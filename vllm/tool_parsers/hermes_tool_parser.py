# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)
from vllm.utils.mistral import is_mistral_tokenizer

logger = init_logger(__name__)


class Hermes2ProToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        if is_mistral_tokenizer(tokenizer):
            logger.error("Detected Mistral tokenizer when using a Hermes model")
            self.model_tokenizer = tokenizer.tokenizer

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
        )
        self.complete_tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.scratch_pad_regex = re.compile(
            r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        self.tool_call_start_token_ids = self.model_tokenizer.encode(
            self.tool_call_start_token, add_special_tokens=False
        )
        self.tool_call_end_token_ids = self.model_tokenizer.encode(
            self.tool_call_end_token, add_special_tokens=False
        )

        self.tool_call_start_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_start_token_ids
        ]

        self.tool_call_end_token_array = [
            self.model_tokenizer.decode([token_id])
            for token_id in self.tool_call_end_token_ids
        ]

        self.buffered_delta_text = ""

    def _extract_complete_tool_call_portion(
        self, text: str, tool_index: int
    ) -> str | None:
        matches = self.complete_tool_call_regex.findall(text)
        if not matches:
            return None
        if 0 <= tool_index < len(matches):
            return matches[tool_index].rstrip()
        return matches[-1].rstrip()

    def _prime_following_tool_state(self, trailing_text: str) -> None:
        if self.tool_call_start_token not in trailing_text:
            return

        next_tool_portion = trailing_text.split(self.tool_call_start_token, 1)[-1]
        self.current_tool_id += 1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool.append("")

        try:
            next_tool_call = partial_json_parser.loads(
                next_tool_portion or "{}",
                Allow.ALL & ~Allow.STR,
            )
            if not isinstance(next_tool_call, dict):
                next_tool_call = {}
        except (
            partial_json_parser.core.exceptions.MalformedJSON,
            json.decoder.JSONDecodeError,
        ):
            next_tool_call = {}

        if len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append(next_tool_call)
        else:
            self.prev_tool_call_arr[self.current_tool_id] = next_tool_call

    # Very simple idea: when encountering tokens like <, tool, _call, >,
    # <, /, tool, _call, >, store them in a buffer.
    # When the last token is encountered, empty the buffer and return it.
    # If a token appears in an incorrect sequence while storing in the buffer,
    # return the preceding buffer along with the token.
    def tool_call_delta_buffer(self, delta_text: str):
        # If the sequence of tool_call_start or tool_call_end tokens is not yet
        # complete, fill the buffer with the token and return "".
        if (
            delta_text in self.tool_call_start_token_array
            or delta_text in self.tool_call_end_token_array
        ):
            # If delta_text is the last token of tool_call_start_token or
            # tool_call_end_token, empty the buffer and return
            # the buffered text + delta_text.
            if (
                delta_text == self.tool_call_start_token_array[-1]
                or delta_text == self.tool_call_end_token_array[-1]
            ):
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                self.buffered_delta_text = self.buffered_delta_text + delta_text
                return ""
        else:
            if self.buffered_delta_text:
                buffered_text = self.buffered_delta_text
                self.buffered_delta_text = ""
                return buffered_text + delta_text
            else:
                return delta_text

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because the tool_call tokens are
            # marked "special" in some models. Since they are skipped
            # prior to the call to the tool parser, it breaks tool calling.
            request.skip_special_tokens = False
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        else:
            try:
                # there are two possible captures - between tags, or between a
                # tag and end-of-string so the result of
                # findall is an array of tuples where one is a function call and
                # the other is None
                function_call_tuples = self.tool_call_regex.findall(model_output)

                # load the JSON, and then use it to build the Function and
                # Tool Call
                raw_function_calls = [
                    json.loads(match[0] if match[0] else match[1])
                    for match in function_call_tuples
                ]
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(
                                function_call["arguments"], ensure_ascii=False
                            ),
                        ),
                    )
                    for function_call in raw_function_calls
                ]

                content = model_output[: model_output.find(self.tool_call_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # 1. All tokens are parsed based on _text, not token_ids.
        # 2. All incoming text data is processed by the tool_call_delta_buffer
        #    function for buffering before being used for parsing.

        delta_text = self.tool_call_delta_buffer(delta_text)
        raw_delta_text = delta_text
        # If the last characters of previous_text
        # match self.buffered_delta_text, remove only the matching part.
        if (
            len(previous_text) >= len(self.buffered_delta_text)
            and previous_text[-len(self.buffered_delta_text) :]
            == self.buffered_delta_text
        ):
            previous_text = previous_text[: -len(self.buffered_delta_text)]
            current_text = previous_text + delta_text

        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)
        # check to see if we should be streaming a tool call - is there a
        if self.tool_call_start_token not in current_text:
            logger.debug("No tool call tokens found!")
            return DeltaMessage(content=delta_text)

        try:
            # figure out where we are in the parsing by counting tool call
            # start & end tags
            prev_tool_start_count = previous_text.count(self.tool_call_start_token)
            prev_tool_end_count = previous_text.count(self.tool_call_end_token)
            cur_tool_start_count = current_text.count(self.tool_call_start_token)
            cur_tool_end_count = current_text.count(self.tool_call_end_token)
            tool_call_portion = None
            text_portion = None
            trailing_tool_text = None
            close_active_tool = (
                self.tool_call_end_token in raw_delta_text
                and prev_tool_start_count > prev_tool_end_count
            )

            # case: if we're generating text, OR rounding out a tool call
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                logger.debug("Generating text content! skipping tool parsing.")
                return DeltaMessage(content=delta_text)

            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                tool_index = (
                    self.current_tool_id
                    if self.current_tool_id >= 0
                    else cur_tool_end_count - 1
                )
                tool_call_portion = self._extract_complete_tool_call_portion(
                    current_text, tool_index
                )
                delta_text, text_portion = raw_delta_text.split(
                    self.tool_call_end_token, 1
                )
                delta_text = delta_text.rstrip()

            # case: if tool open & close tag counts don't match, we're doing
            # imaginary "else" block here
            # something with tools with this diff.
            # flags for partial JSON parting. exported constants from
            # "Allow" are handled via BIT MASK
            flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

            # case -- the current tool call is being closed.
            if close_active_tool or (
                cur_tool_start_count == cur_tool_end_count
                and cur_tool_end_count >= prev_tool_end_count
            ):
                emitted_content = None
                closed_tool_id = self.current_tool_id
                if text_portion:
                    if self.tool_call_start_token in text_portion:
                        emitted_content = (
                            text_portion.split(self.tool_call_start_token, 1)[0] or None
                        )
                        trailing_tool_text = text_portion
                    else:
                        emitted_content = text_portion or None

                if self.prev_tool_call_arr is None or len(self.prev_tool_call_arr) == 0:
                    if (
                        self.current_tool_id >= 0
                        and self.current_tool_name_sent
                        and tool_call_portion
                    ):
                        try:
                            current_tool_call = json.loads(tool_call_portion)
                        except json.decoder.JSONDecodeError:
                            logger.debug("unable to parse closing tool call JSON")
                            return None

                        arguments = current_tool_call.get("arguments")
                        arguments_json = (
                            arguments
                            if isinstance(arguments, str)
                            else json.dumps(arguments, ensure_ascii=False)
                        )
                        diff = arguments_json
                        self.prev_tool_call_arr.append(current_tool_call)
                        self.streamed_args_for_tool[self.current_tool_id] = (
                            arguments_json
                        )
                        if trailing_tool_text:
                            self._prime_following_tool_state(trailing_tool_text)
                        if not diff and emitted_content is None:
                            return None
                        return DeltaMessage(
                            content=emitted_content,
                            tool_calls=[
                                DeltaToolCall(
                                    index=closed_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=diff or None
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                            if diff
                            else [],
                        )

                    if tool_call_portion and cur_tool_start_count > prev_tool_start_count:
                        try:
                            current_tool_call = json.loads(tool_call_portion)
                        except json.decoder.JSONDecodeError:
                            logger.debug("unable to parse single-chunk tool call JSON")
                            return None

                        function_name = current_tool_call.get("name")
                        arguments = current_tool_call.get("arguments")
                        if function_name and arguments is not None:
                            arguments_json = (
                                arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments, ensure_ascii=False)
                            )
                            self.current_tool_id += 1
                            self.current_tool_name_sent = True
                            self.prev_tool_call_arr.append(current_tool_call)
                            self.streamed_args_for_tool.append(arguments_json)
                            prefix_content = (
                                raw_delta_text.split(self.tool_call_start_token, 1)[0]
                                or ""
                            )
                            content = prefix_content + (emitted_content or "")
                            if trailing_tool_text:
                                self._prime_following_tool_state(trailing_tool_text)
                            return DeltaMessage(
                                content=content or None,
                                tool_calls=[
                                    DeltaToolCall(
                                        index=closed_tool_id + 1,
                                        type="function",
                                        id=make_tool_call_id(),
                                        function=DeltaFunctionCall(
                                            name=function_name,
                                            arguments=arguments_json,
                                        ).model_dump(exclude_none=True),
                                    )
                                ],
                            )

                    logger.debug("attempting to close tool call, but no tool call")
                    return None
                if tool_call_portion is None:
                    return None

                try:
                    current_tool_call = json.loads(tool_call_portion)
                except json.decoder.JSONDecodeError:
                    logger.debug("unable to parse closing tool call JSON")
                    return None

                arguments = current_tool_call.get("arguments")
                arguments_json = (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False)
                )
                actual_streamed = self.streamed_args_for_tool[self.current_tool_id]
                if arguments_json.startswith(actual_streamed):
                    diff = arguments_json[len(actual_streamed) :]
                else:
                    diff = arguments_json.replace(actual_streamed, "", 1)

                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
                self.streamed_args_for_tool[self.current_tool_id] = arguments_json
                if trailing_tool_text:
                    self._prime_following_tool_state(trailing_tool_text)
                if not diff and emitted_content is None:
                    return None

                logger.debug(
                    "Finishing tool and found diff that had not "
                    "been streamed yet: %s",
                    diff,
                )
                return DeltaMessage(
                    content=emitted_content,
                    tool_calls=[
                        DeltaToolCall(
                            index=closed_tool_id,
                            function=DeltaFunctionCall(arguments=diff or None).model_dump(
                                exclude_none=True
                            ),
                        )
                    ]
                    if diff
                    else [],
                )

            # case -- we're starting a new tool call
            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count > prev_tool_start_count
            ):
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]

                text_portion = None

                # set cursors and state appropriately
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            # case -- we're updating an existing tool call
            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count == prev_tool_start_count
            ):
                # get the portion of the text that's the tool call
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                text_portion = None

            # case -- otherwise we're just generating text
            else:
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                delta = DeltaMessage(tool_calls=[], content=text)
                return delta

            try:
                current_tool_call = (
                    partial_json_parser.loads(tool_call_portion or "{}", flags)
                    if tool_call_portion
                    else None
                )
                logger.debug("Parsed tool call %s", current_tool_call)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None
            except json.decoder.JSONDecodeError:
                logger.debug("unable to parse JSON")
                return None

            if current_tool_call is None:
                return None

            # case - we haven't sent the tool name yet. If it's available, send
            #   it. otherwise, wait until it's available.
            if not self.current_tool_name_sent:
                function_name: str | None = current_tool_call.get("name")
                if function_name:
                    self.current_tool_name_sent = True
                    if len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append(current_tool_call)
                    else:
                        self.prev_tool_call_arr[self.current_tool_id] = (
                            current_tool_call
                        )
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=make_tool_call_id(),
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                else:
                    return None
            # case -- otherwise, send the tool call delta

            # if the tool call portion is None, send the delta as text
            if tool_call_portion is None:
                # if there's text but not tool calls, send that -
                # otherwise None to skip chunk
                delta = (
                    DeltaMessage(content=delta_text)
                    if text_portion is not None
                    else None
                )
                return delta

            # now, the nitty-gritty of tool calls
            # now we have the portion to parse as tool call.

            if current_tool_call is None:
                return None

            logger.debug(
                "Trying to parse current tool call with ID %s", self.current_tool_id
            )

            # if we're starting a new tool call, push an empty object in as
            #   a placeholder for the arguments
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            # main logic for tool parsing here - compare prev. partially-parsed
            #   JSON to the current partially-parsed JSON
            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )
            assert current_tool_call is not None
            cur_arguments = current_tool_call.get("arguments")

            logger.debug("diffing old arguments: %s", prev_arguments)
            logger.debug("against new ones: %s", cur_arguments)

            # case -- no arguments have been created yet. skip sending a delta.
            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
                delta = None

            # case -- prev arguments are defined, but non are now.
            #   probably impossible, but not a fatal error - just keep going
            elif not cur_arguments and prev_arguments:
                logger.error(
                    "should be impossible to have arguments reset "
                    "mid-call. skipping streaming anything."
                )
                delta = None

            # case -- we now have the first info about arguments available from
            #   autocompleting the JSON
            elif cur_arguments and not prev_arguments:
                # extract the content after {"name": ..., "arguments":
                #   directly from tool_call_portion as cur_arguments_json,
                #   since cur_arguments may differ from the original text
                #   due to partial JSON parsing
                #   for example, tool_call_portion =
                #     {"name": "search", "arguments": {"search_request": {"
                #   but cur_arguments =
                #     {"search_request": {}}
                function_name = current_tool_call.get("name")
                match = re.search(
                    r'\{"name":\s*"'
                    + re.escape(function_name)
                    + r'"\s*,\s*"arguments":\s*(.*)',
                    tool_call_portion.strip(),
                    re.DOTALL,
                )
                if match:
                    cur_arguments_json = match.group(1)
                else:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)

                logger.debug("finding %s in %s", delta_text, cur_arguments_json)

                # get the location where previous args differ from current.
                if delta_text not in cur_arguments_json:
                    return None
                args_delta_start_loc = cur_arguments_json.rindex(delta_text) + len(
                    delta_text
                )

                # use that to find the actual delta
                arguments_delta = cur_arguments_json[:args_delta_start_loc]
                logger.debug("First tokens in arguments received: %s", arguments_delta)

                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=arguments_delta
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] += arguments_delta

            # last case -- we have an update to existing arguments.
            elif cur_arguments and prev_arguments:
                # judge whether the tool_call_portion is a complete JSON
                try:
                    json.loads(tool_call_portion)
                    is_complete_json = True
                except Exception:
                    is_complete_json = False

                # if the delta_text ends with a '}' and tool_call_portion is a
                #   complete JSON, then the last '}' does not belong to the
                #   arguments, so we should trim it off
                if (
                    isinstance(delta_text, str)
                    and len(delta_text.rstrip()) >= 1
                    and delta_text.rstrip()[-1] == "}"
                    and is_complete_json
                ):
                    delta_text = delta_text.rstrip()[:-1]

                logger.debug("got diff %s", delta_text)

                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=delta_text).model_dump(
                                exclude_none=True
                            ),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] += delta_text

            # handle saving the state for the current tool into
            # the "prev" list for use in diffing for the next iteration
            assert isinstance(current_tool_call, dict)
            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.
