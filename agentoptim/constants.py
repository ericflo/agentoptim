"""Constants used throughout AgentOptim."""

# Maximum number of questions allowed per EvalSet
MAX_QUESTIONS_PER_EVALSET = 100

# Maximum number of EvalSets allowed
MAX_EVALSETS = 100

# Maximum number of evaluation runs to store
MAX_EVAL_RUNS = 500

# Default timeout values
DEFAULT_API_TIMEOUT_SECONDS = 120  # 2 minutes

# API base URLs
DEFAULT_LOCAL_API_BASE = "http://localhost:1234/v1"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"

# Default models for different providers
DEFAULT_LOCAL_MODEL = "meta-llama-3.1-8b-instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"

# Default pagination settings
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100