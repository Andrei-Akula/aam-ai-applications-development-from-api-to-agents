from t2_llms_output_tuning._clients.anthropic_client import AnthropicAIClient
from t2_llms_output_tuning._main import run

# TODO 1: temperature — controls randomness. Range: 0.0-1.0, default: 1.0
#  Lower = more deterministic, higher = more creative
#  Query: "Give me a name for a coffee shop"
#  Try: temperature=0.0 vs temperature=1.0, compare outputs
run(
    client=AnthropicAIClient('claude-sonnet-4-5'),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    user_input="Give me a name for a coffee shop",
    temperature=0.0, # Adjust temperature for randomness
    max_tokens=20, # Adjust max_tokens for response length
)
run(
    client=AnthropicAIClient('claude-sonnet-4-5'),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    user_input="Give me a name for a coffee shop",
    temperature=1.0, # Adjust temperature for randomness
    max_tokens=20, # Adjust max_tokens for response length
)

# TODO 2: top_p — nucleus sampling, keeps tokens within cumulative probability. Range: 0.0-1.0, default: 1.0 (disabled)
#  Lower = fewer token choices, more focused output
#  Query: "List 5 alternative uses for a paperclip"
#  Try: top_p=0.1 vs top_p=0.9
run(
    client=AnthropicAIClient('claude-sonnet-4-5'),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    user_input="List 5 alternative uses for a paperclip",
    top_p=0.1, # Adjust top_p for nucleus sampling
    max_tokens=20, # Adjust max_tokens for response length
)

run(
    client=AnthropicAIClient('claude-sonnet-4-5'),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    user_input="List 5 alternative uses for a paperclip",
    top_p=0.9, # Adjust top_p for nucleus sampling
    max_tokens=20, # Adjust max_tokens for response length
)


# TODO 3: top_k — limits token selection to top K candidates. Default: not set (disabled)
#  Lower = fewer choices per token, more predictable
#  Query: "Write a one-sentence story about a robot"
#  Try: top_k=1 vs top_k=50
run(
    client=AnthropicAIClient('claude-sonnet-4-5'),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    user_input="Write a one-sentence story about a robot",
    top_k=1, # Adjust top_k for limiting token selection
    max_tokens=20, # Adjust max_tokens for response length
)
run(
    client=AnthropicAIClient('claude-sonnet-4-5'),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    user_input="Write a one-sentence story about a robot",
    top_k=50, # Adjust top_k for limiting token selection
    max_tokens=20, # Adjust max_tokens for response length
)


# TODO 4: stop_sequences — list of strings that stop generation when encountered
#  Query: "Count from 1 to 20, comma separated"
#  Try: stop_sequences=["10"] — generation stops before reaching 10
run(
    client=AnthropicAIClient('claude-sonnet-4-5'),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    user_input="Count from 1 to 20, comma separated",
    stop_sequences=["10"], # Stop generation when encountering "10"
    max_tokens=20, # Adjust max_tokens for response length
)


# TODO 5: output_config — enforce structured JSON output
#  Query: "List 3 programming languages with their year of creation"
#  Try: output_config={"format":{"type": "json_schema", "schema": {"type": "object", "additionalProperties": False, "properties": {"languages": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},"required": ["name", "year"]}}}}}}
run(
    client=AnthropicAIClient('claude-sonnet-4-5'),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    user_input="List 3 programming languages with their year of creation",
    output_config={"format":{"type": "json_schema", "schema": {"type": "object", "additionalProperties": False, "properties": {"languages": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},"required": ["name", "year"]}}}}}},
    max_tokens=200, # Adjust max_tokens for response length
)

# TODO 6: thinking — enables extended thinking (chain-of-thought). Requires budget_tokens param
#  Model reasons step-by-step before answering. Needs max_tokens > budget_tokens
#  Query: "How many r's are in the word strawberry?"
#  Try: thinking={"type": "enabled", "budget_tokens": 5000}, max_tokens=8000
run(
    client=AnthropicAIClient('claude-sonnet-4-5'),
    print_request=True, # Switch to False if you do not want to see the request in console
    print_only_content=False, # Switch to True if you want to see only content from response
    user_input="How many r's are in the word strawberry?",
    thinking={"type": "enabled", "budget_tokens": 5000},
    max_tokens=8000, # Adjust max_tokens for response length
)