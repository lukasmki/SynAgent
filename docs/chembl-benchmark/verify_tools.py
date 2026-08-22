"""Prove the tools are genuinely invoked, not just named in prose.

Reports, for one route, every structural ToolCallPart (name + arguments the
model actually emitted) and every ToolReturnPart (what the tool returned).
Text matching on the rendered reply cannot distinguish "the model called
fix_step" from "the model wrote the words fix_step", so this reads the typed
message parts instead.
"""
import asyncio, csv, json, sys
sys.path.insert(0, ".")
csv.field_size_limit(10**7)
from run_bench import classify
from pydantic_ai.messages import ModelRequest, ModelResponse
from synagent.synagent import get_agent


def dump(messages, label):
    print(f"\n{'='*68}\n{label}\n{'='*68}")
    for m in messages:
        if isinstance(m, ModelResponse):
            for p in m.parts:
                if getattr(p, "part_kind", "") == "tool-call":
                    args = p.args
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except Exception: pass
                    print(f"  ->  CALL   {p.tool_name}({json.dumps(args)[:160]})")
                    print(f"             tool_call_id={getattr(p,'tool_call_id','?')}")
        elif isinstance(m, ModelRequest):
            for p in m.parts:
                if getattr(p, "part_kind", "") == "tool-return":
                    c = p.content
                    if hasattr(c, "model_dump"):
                        c = c.model_dump()
                    s = json.dumps(c, default=str) if not isinstance(c, str) else c
                    print(f"  <-  RETURN {p.tool_name}: {s[:300]}")
                elif getattr(p, "part_kind", "") == "retry-prompt":
                    print(f"  !!  RETRY  {str(getattr(p,'content',''))[:200]}")


async def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rows = [r for r in csv.DictReader(open("synllama-raw-failed.csv", encoding="utf-8"))
            if len(r["response"]) <= 1400]
    import random; random.Random(42).shuffle(rows)
    row = rows[idx]
    print(f"ROUTE #{idx}  target={row['smiles'][:50]}")
    print(f"strict verdict BEFORE: {classify(row['response'])}")

    agent = get_agent("deepseek-chat", provider="deepseek")
    r1 = await agent.run(f"Validate this synthesis route and report every step:\n\n{row['response']}")
    dump(r1.all_messages(), "TURN 1  — validate")
    r2 = await agent.run("Now fix the failed steps in that route.",
                         message_history=r1.all_messages())
    dump(r2.all_messages()[len(r1.all_messages()):], "TURN 2 — fix (new messages only)")

asyncio.run(main())
