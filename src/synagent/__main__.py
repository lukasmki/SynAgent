import json
from pathlib import Path

import polars as pl
import typer
from tqdm import tqdm

from synagent.agents import get_agent
from synagent.validate import (
    ProductsError,
    ReactantError,
    SmilesError,
    validate,
    ReactionError,
)

app = typer.Typer()


@app.command(name="eval")
def eval(file: Path, sampling: str = "all"):
    df = pl.read_csv(file.resolve())
    if sampling == "all":
        pass
    else:
        df = df.filter(pl.col("sampling_params") == sampling)

    ntotal: int = df.height
    errors = {
        "json": 0,
        "smiles": 0,
        "reactant": 0,
        "reaction": 0,
        "product": 0,
        "other": 0,
    }
    for row in df.rows(named=True):
        try:
            validate(row["response"])
        except json.JSONDecodeError:
            errors["json"] += 1
        except SmilesError:
            errors["smiles"] += 1
        except ReactantError:
            errors["reactant"] += 1
        except ReactionError:
            errors["reaction"] += 1
        except ProductsError:
            errors["product"] += 1
        except Exception:
            errors["other"] += 1

    total_errors = sum(errors.values())
    print(
        f"Valid percentatge: {ntotal - total_errors}/{ntotal}, {(ntotal - total_errors) / ntotal * 100:3.2f}%"
    )
    for err, val in errors.items():
        print(
            f"{err}: {val}/{total_errors}, {val / total_errors * 100:3.2f}% of errors"
        )


@app.command(name="run")
def run(file: Path, agent_name: str, output: Path | None = None):
    if output is None:
        output = file.with_name("output").with_suffix(".jsonl")

    df = pl.read_csv(file.resolve())
    agent = get_agent(agent_name)

    for row in tqdm(df.rows(named=True)):
        prompt = ",".join(row.values())
        result = agent.run_sync(prompt)

        json_bytes = result.all_messages_json()
        with output.open("ab") as f:
            f.write(json_bytes + b"\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
