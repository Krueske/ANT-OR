"""
Token consumption statistics tool.
Uses tokenizer to count tokens for LLM input/output.
"""
import threading

try:
    from qwen_tokenizer import qwen_tokenizer
    HAS_QWEN_TOKENIZER = True
except ImportError:
    HAS_QWEN_TOKENIZER = False
    try:
        import tiktoken
        HAS_TIKTOKEN = True
    except ImportError:
        HAS_TIKTOKEN = False


_tokenizer = None
_tokenizer_lock = threading.Lock()


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                if HAS_QWEN_TOKENIZER:
                    _tokenizer = qwen_tokenizer.QwenTokenizer(vocab_file="utils/qwen.tiktoken")
                elif HAS_TIKTOKEN:
                    _tokenizer = tiktoken.get_encoding("cl100k_base")
                else:
                    _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in a single text."""
    if not text:
        return 0
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return len(text) // 4 + 1
    return len(tokenizer.encode(text))


def count_messages_tokens(messages: list) -> int:
    """Count input tokens in messages format (approximate, including role markers)."""
    total = 0
    for msg in messages:
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    total += count_tokens(item.get("text", ""))
    total += 2
    return total


class TokenCounter:
    """Thread-safe token consumption accumulator."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.step_stats = {}  # step_name -> {"input": int, "output": int, "calls": int}

    def record(self, input_messages: list, output_text: str, step_name: str = "unknown"):
        """Record token consumption for one LLM call."""
        input_tokens = count_messages_tokens(input_messages)
        output_tokens = count_tokens(output_text)

        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            if step_name not in self.step_stats:
                self.step_stats[step_name] = {"input": 0, "output": 0, "calls": 0}
            self.step_stats[step_name]["input"] += input_tokens
            self.step_stats[step_name]["output"] += output_tokens
            self.step_stats[step_name]["calls"] += 1

    @property
    def total_tokens(self):
        return self.total_input_tokens + self.total_output_tokens

    def get_summary(self) -> dict:
        """Return summary statistics."""
        with self._lock:
            summary = {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "by_step": {}
            }
            for step_name, stats in self.step_stats.items():
                summary["by_step"][step_name] = {
                    "input_tokens": stats["input"],
                    "output_tokens": stats["output"],
                    "total_tokens": stats["input"] + stats["output"],
                    "calls": stats["calls"]
                }
            return summary

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print("Token Usage Statistics")
        print(f"{'='*60}")
        print(f"  Total input tokens:   {summary['total_input_tokens']:>10,}")
        print(f"  Total output tokens:  {summary['total_output_tokens']:>10,}")
        print(f"  Total tokens:         {summary['total_tokens']:>10,}")
        print(f"{'-'*60}")
        print(f"  {'Step':<30} {'Calls':>8} {'Input':>10} {'Output':>10} {'Total':>10}")
        print(f"{'-'*60}")
        for step_name, stats in summary["by_step"].items():
            print(f"  {step_name:<30} {stats['calls']:>8} {stats['input_tokens']:>10,} {stats['output_tokens']:>10,} {stats['total_tokens']:>10,}")
        print(f"{'='*60}")
