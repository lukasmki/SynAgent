import json
from pathlib import Path

import polars as pl
import typer
import uvicorn
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from synagent.agents import get_agent
from synagent.validate import (
    ProductsError,
    ReactantError,
    ReactionError,
    SmilesError,
    validate,
)

from synagent.chemspacetool import ChemspaceDeps
from synagent.tokenmanager import ChemspaceTokenManager

app = typer.Typer()


@app.command(name="eval")
def eval(file: Path, output_dir: str = "data/", sampling: str = "all"):
    df = pl.read_csv(file.resolve())
    if sampling == "all":
        pass
    else:
        df = df.filter(pl.col("sampling_params") == sampling)

    valid = []
    failed = []

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
        success = False
        try:
            validate(row["response"])
            success = True
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

        if success:
            valid.append(row)
        else:
            failed.append(row)

    total_errors = sum(errors.values())
    print(
        f"Valid percentatge: {ntotal - total_errors}/{ntotal}, {(ntotal - total_errors) / ntotal * 100:3.2f}%"
    )
    for err, val in errors.items():
        print(
            f"{err}: {val}/{total_errors}, {val / total_errors * 100:3.2f}% of errors"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fail_df = pl.DataFrame(failed)
    fail_df.write_csv(output / "synllama-raw-failed.csv")
    valid_df = pl.DataFrame(valid)
    valid_df.write_csv(output / "synllama-raw-valid.csv")


@app.command(name="run")
def run(file: Path, agent_name: str, output: Path | None = None):
    if output is None:
        output = file.with_name("output").with_suffix(".jsonl")

    df = pl.read_csv(file.resolve())
    agent = get_agent(agent_name)

    deps = None
    if agent_name.lower() == "chemspace":
        mgr = ChemspaceTokenManager()
        deps = ChemspaceDeps(mgr=mgr)

    for row in tqdm(df.rows(named=True)):
        prompt = ",".join(str(v) for v in row.values())

        if deps is not None:
            result = agent.run_sync(prompt, deps=deps)
        else:
            result = agent.run_sync(prompt)

        json_bytes = result.all_messages_json()
        with output.open("ab") as f:
            f.write(json_bytes + b"\n")


@app.command(name="serve")
def serve(
    agent_name: str,
    host: str = "127.0.0.1",
    port: int = 8000,
):
    agent = get_agent(agent_name)

    if agent_name.lower() in {"chemspace", "master"}:
        mgr = ChemspaceTokenManager()
        deps = ChemspaceDeps(mgr=mgr)
        uvicorn.run(agent.to_web(deps=deps), host=host, port=port)
    else:
        uvicorn.run(agent.to_web(), host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()