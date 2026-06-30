from dataclasses import dataclass, field

HAZARD_WEIGHTS: dict[str, float] = {
    # Acute toxicity
    "H300": 1.00,  # Fatal if swallowed
    "H310": 1.00,  # Fatal in contact with skin
    "H330": 1.00,  # Fatal if inhaled
    "H301": 0.82,  # Toxic if swallowed
    "H311": 0.82,  # Toxic in contact with skin
    "H331": 0.82,  # Toxic if inhaled
    "H302": 0.58,  # Harmful if swallowed
    "H312": 0.58,  # Harmful in contact with skin
    "H332": 0.58,  # Harmful if inhaled
    # Corrosion / irritation / sensitization
    "H314": 0.72,  # Causes severe skin burns and eye damage
    "H315": 0.30,  # Causes skin irritation
    "H318": 0.62,  # Causes serious eye damage
    "H319": 0.25,  # Causes serious eye irritation
    "H317": 0.36,  # May cause an allergic skin reaction
    "H334": 0.80,  # May cause allergy or asthma symptoms or breathing difficulties if inhaled
    # CMR
    "H340": 0.93,  # May cause genetic defects
    "H341": 0.68,  # Suspected of causing genetic defects
    "H350": 0.95,  # May cause cancer
    "H351": 0.72,  # Suspected of causing cancer
    "H360": 0.93,  # May damage fertility or the unborn child
    "H361": 0.70,  # Suspected of damaging fertility or the unborn child
    "H362": 0.45,  # May cause harm to breast-fed children
    # STOT / aspiration
    "H370": 0.88,  # Causes damage to organs
    "H371": 0.63,  # May cause damage to organs
    "H335": 0.35,  # May cause respiratory irritation
    "H336": 0.35,  # May cause drowsiness or dizziness
    "H372": 0.90,  # Causes damage to organs through prolonged/repeated exposure
    "H373": 0.66,  # May cause damage to organs through prolonged/repeated exposure
    "H304": 0.83,  # May be fatal if swallowed and enters airways
    "H305": 0.50,  # May be harmful if swallowed and enters airways
    # Explosives / flammability / reactivity
    "H200": 1.00,  # Unstable explosive
    "H201": 0.98,  # Explosive; mass explosion hazard
    "H202": 0.96,  # Explosive; severe projection hazard
    "H203": 0.92,  # Explosive; fire, blast or projection hazard
    "H204": 0.72,  # Fire or projection hazard
    "H205": 0.76,  # May mass explode in fire
    "H220": 0.72,  # Extremely flammable gas
    "H221": 0.48,  # Flammable gas
    "H224": 0.70,  # Extremely flammable liquid and vapor
    "H225": 0.55,  # Highly flammable liquid and vapor
    "H226": 0.38,  # Flammable liquid and vapor
    "H227": 0.22,  # Combustible liquid
    "H228": 0.45,  # Flammable solid
    "H240": 0.95,  # Heating may cause an explosion
    "H241": 0.85,  # Heating may cause a fire or explosion
    "H242": 0.68,  # Heating may cause a fire
    "H250": 0.90,  # Catches fire spontaneously if exposed to air
    "H251": 0.76,  # Self-heating; may catch fire
    "H252": 0.45,  # Self-heating in large quantities
    "H260": 0.86,  # In contact with water releases flammable gases which may ignite spontaneously
    "H261": 0.68,  # In contact with water releases flammable gases
    "H270": 0.78,  # May cause or intensify fire; oxidizer
    "H271": 0.90,  # May cause fire or explosion; strong oxidizer
    "H272": 0.46,  # May intensify fire; oxidizer
    "H230": 0.95,  # May react explosively even in absence of air
    "H231": 0.88,  # May react explosively even in absence of air at elevated pressure and/or temperature
    # Environmental
    "H400": 0.40,
    "H410": 0.46,
    "H411": 0.33,
    "H412": 0.20,
    "H413": 0.10,
}


RED_FLAG_CODES = {
    "H200",
    "H201",
    "H202",
    "H203",
    "H205",
    "H240",
    "H271",
    "H250",
    "H300",
    "H310",
    "H330",
}


@dataclass
class CompoundHazard:
    name: str
    hazard_codes: list[str]
    matched_weights: list[float] = field(default_factory=list)
    compound_hazard: float = 0.0
    red_flag: bool = False
