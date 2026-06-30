from pathlib import Path

from FPSim2 import FPSim2Engine, ReactionEngine
from FPSim2.io import create_db_file, create_reaction_db_file

# reactions
root = Path(__file__).parent

if not (root / "reactions.h5").exists():
    create_reaction_db_file(
        rxns_source=str(root / "115_rxn_templates.sma"),
        filename=str(
            root / "reactions.h5",
        ),
        fp_type="RDKitPattern",
        store_strings=True,
    )

rxn_engine = ReactionEngine(str(root / "reactions.h5"))
results = rxn_engine.similarity(
    "[c:1]-B(O)O.[Cl,Br,I][c:2]>>[c:1]-[c:2]", threshold=0.6, metric="cosine"
)
print(len(results))
print(rxn_engine.get_strings(results))

# building blocks
if not (root / "building_blocks.h5").exists():
    create_db_file(
        mols_source=str(root / "enamine-rdbb-us.sdf"),
        filename=str(root / "building_blocks.h5"),
        mol_format=None,
        fp_type="RDKitPattern",
        mol_id_prop=None,
        store_strings=True,
    )

mol_engine = FPSim2Engine(root / "building_blocks.h5")
results = mol_engine.substructure("[c:1]-B(O)O.[Cl,Br,I][c:2]", mol_format="smarts")
print(len(results))
