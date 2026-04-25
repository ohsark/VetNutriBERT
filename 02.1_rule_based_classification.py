import warnings
import re
from pathlib import Path
from typing import List, Optional

import medspacy
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from medspacy.context import ConTextRule
from medspacy.ner import TargetRule
from medspacy.visualization import visualize_ent  # noqa: F401
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

DATA_PATH = Path("./input/strat_sample_v1_cleaned_df.xlsx")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ground truth column in DATA_PATH
ANNOTATION_COLUMN = "Annotation"

# Common pet names that overlap with food terms
FOOD_PET_NAMES = {
    "salmon",
    "chicken",
    "rice",
    "cookie",
    "biscuit",
    "ginger",
    "pepper",
    "cinnamon",
    "olive",
    "peaches",
    "honey",
    "sugar",
    "cocoa",
    "mocha",
    "brownie",
    "muffin",
    "pumpkin",
    "bean",
    "pickles",
    "chip",
    "pretzel",
    "noodle",
    "waffle",
    "taco",
    "nacho",
    "oreo",
    "snickers",
    "twix",
    "brandy",
    "whiskey",
    "whisky",
    "guinness",
    "porter",
    "stout",
    "coco",
    "choco",
    "latte",
    "espresso",
    "chai",
    "matcha",
    "duck",
    "lamb",
}


def detect_pet_name(text: str) -> Optional[str]:
    """Detect likely pet names that collide with diet entities."""
    patterns = [
        r"patient\s*name[:\s]+[\"\']?(\w+)[\"\']?",
        r"\*+\s*patient\s*name[:\s]+[\"\']?(\w+)[\"\']?",
        r"name[:\s]+[\"\']?(\w+)[\"\']?",
        r"(?:hx|history)[:\s]+(\w+)\s+(?:is|was|presented|has)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            potential_name = match.group(1).lower()
            if potential_name in FOOD_PET_NAMES:
                return potential_name
    return None


def mask_pet_name(text: str, pet_name: str) -> str:
    """Mask detected pet names so they do not trigger DIET_ITEM/DIET_BRAND rules."""
    return re.sub(rf"\b{re.escape(pet_name)}\b", "__PETNAME__", text, flags=re.IGNORECASE)


def build_target_rules() -> List[TargetRule]:
    # 1. EATING STATUS
    eating_yes_patterns = [
        TargetRule("eating and drinking normally", "EATING_YES", pattern=[{"LOWER": "eating"}, {"LOWER": "and"}, {"LOWER": "drinking"}, {"LOWER": "normally"}]),
        TargetRule("eating and drinking well", "EATING_YES", pattern=[{"LOWER": "eating"}, {"LOWER": "and"}, {"LOWER": "drinking"}, {"LOWER": "well"}]),
        TargetRule("eating well", "EATING_YES", pattern=[{"LOWER": "eating"}, {"LOWER": "well"}]),
        TargetRule("eating normally", "EATING_YES", pattern=[{"LOWER": "eating"}, {"LOWER": "normally"}]),
        TargetRule("eating fine", "EATING_YES", pattern=[{"LOWER": "eating"}, {"LOWER": "fine"}]),
        TargetRule("eating ok", "EATING_YES", pattern=[{"LOWER": "eating"}, {"LOWER": "ok"}]),
        TargetRule("good appetite", "EATING_YES", pattern=[{"LOWER": "good"}, {"LOWER": "appetite"}]),
        TargetRule("normal appetite", "EATING_YES", pattern=[{"LOWER": "normal"}, {"LOWER": "appetite"}]),
        TargetRule("appetite ok", "EATING_YES", pattern=[{"LOWER": "appetite"}, {"LOWER": "ok"}]),
        TargetRule("appetite remains", "EATING_YES", pattern=[{"LOWER": "appetite"}, {"LOWER": "remains"}]),
        TargetRule("appetite good", "EATING_YES", pattern=[{"LOWER": "appetite"}, {"LOWER": "good"}]),
        TargetRule("appetite is good", "EATING_YES", pattern=[{"LOWER": "appetite"}, {"LOWER": "is"}, {"LOWER": "good"}]),
        TargetRule("has appetite", "EATING_YES", pattern=[{"LOWER": "has"}, {"LOWER": {"IN": ["appetite", "his", "her"]}}, {"LOWER": "appetite"}]),
        TargetRule("owner reports eating well", "EATING_YES", pattern=[{"LOWER": "owner"}, {"LOWER": "reports"}, {"LOWER": "eating"}, {"LOWER": "well"}]),
        TargetRule("feeding well", "EATING_YES", pattern=[{"LOWER": "feeding"}, {"LOWER": "well"}]),
        TargetRule("fed well", "EATING_YES", pattern=[{"LOWER": "fed"}, {"LOWER": "well"}]),
        TargetRule("food motivated", "EATING_YES", pattern=[{"LOWER": "food"}, {"LOWER": "motivated"}]),
        TargetRule("loves food", "EATING_YES", pattern=[{"LOWER": "loves"}, {"LOWER": "food"}]),
        TargetRule("keen on food", "EATING_YES", pattern=[{"LOWER": "keen"}, {"LOWER": "on"}, {"LOWER": "food"}]),
        TargetRule("happily eating", "EATING_YES", pattern=[{"LOWER": "happily"}, {"LOWER": "eating"}]),
        TargetRule("happy eating", "EATING_YES", pattern=[{"LOWER": "happy"}, {"LOWER": "eating"}]),
    ]

    eating_no_patterns = [
        TargetRule("not eating", "EATING_NO", pattern=[{"LOWER": "not"}, {"LOWER": "eating"}]),
        TargetRule("not really eating", "EATING_NO", pattern=[{"LOWER": "not"}, {"LOWER": "really"}, {"LOWER": "eating"}]),
        TargetRule("not eating well", "EATING_NO", pattern=[{"LOWER": "not"}, {"LOWER": "eating"}, {"LOWER": "well"}]),
        TargetRule("not eating anything", "EATING_NO", pattern=[{"LOWER": "not"}, {"LOWER": "eating"}, {"LOWER": "anything"}]),
        TargetRule("stopped eating", "EATING_NO", pattern=[{"LOWER": "stopped"}, {"LOWER": "eating"}]),
        TargetRule("loss of appetite", "EATING_NO", pattern=[{"LOWER": "loss"}, {"LOWER": "of"}, {"LOWER": "appetite"}]),
        TargetRule("lost appetite", "EATING_NO", pattern=[{"LOWER": "lost"}, {"LOWER": {"IN": ["her", "his", "their", "appetite"]}}, {"LOWER": {"IN": ["appetite", "a"]}}]),
        TargetRule("reduced appetite", "EATING_NO", pattern=[{"LOWER": "reduced"}, {"LOWER": "appetite"}]),
        TargetRule("decreased appetite", "EATING_NO", pattern=[{"LOWER": "decreased"}, {"LOWER": "appetite"}]),
        TargetRule("poor appetite", "EATING_NO", pattern=[{"LOWER": "poor"}, {"LOWER": "appetite"}]),
        TargetRule("no appetite", "EATING_NO", pattern=[{"LOWER": "no"}, {"LOWER": "appetite"}]),
        TargetRule("losing appetite", "EATING_NO", pattern=[{"LOWER": "losing"}, {"LOWER": "appetite"}]),
        TargetRule("appetite is less", "EATING_NO", pattern=[{"LOWER": "appetite"}, {"LOWER": "is"}, {"LOWER": "less"}]),
        TargetRule("appetite slightly reduced", "EATING_NO", pattern=[{"LOWER": "appetite"}, {"LOWER": "slightly"}, {"LOWER": "reduced"}]),
        TargetRule("appetite has been on and off", "EATING_NO", pattern=[{"LOWER": "appetite"}, {"LOWER": "has"}, {"LOWER": "been"}, {"LOWER": "on"}, {"LOWER": "and"}, {"LOWER": "off"}]),
        TargetRule("inappetence", "EATING_NO"),
        TargetRule("anorexia", "EATING_NO"),
        TargetRule("anorexic", "EATING_NO"),
        TargetRule("fussy eater", "EATING_NO", pattern=[{"LOWER": "fussy"}, {"LOWER": "eater"}]),
        TargetRule("fussy with eating", "EATING_NO", pattern=[{"LOWER": "fussy"}, {"LOWER": "with"}, {"LOWER": "eating"}]),
        TargetRule("not too keen on", "EATING_NO", pattern=[{"LOWER": "not"}, {"LOWER": "too"}, {"LOWER": "keen"}, {"LOWER": "on"}]),
        TargetRule("not keen on", "EATING_NO", pattern=[{"LOWER": "not"}, {"LOWER": "keen"}, {"LOWER": "on"}]),
        TargetRule("not interested in eating", "EATING_NO", pattern=[{"LOWER": "not"}, {"LOWER": "interested"}, {"LOWER": "in"}, {"LOWER": "eating"}]),
        TargetRule("only eating if hand fed", "EATING_NO", pattern=[{"LOWER": "only"}, {"LOWER": "eating"}, {"LOWER": "if"}, {"LOWER": "hand"}, {"LOWER": "fed"}]),
        TargetRule("eating out of hands", "EATING_NO", pattern=[{"LOWER": "eating"}, {"LOWER": "out"}, {"LOWER": "of"}, {"LOWER": "hands"}]),
        TargetRule("hand fed", "EATING_NO", pattern=[{"LOWER": "hand"}, {"LOWER": "fed"}]),
        TargetRule("partially eating", "EATING_NO", pattern=[{"LOWER": "partially"}, {"LOWER": "eating"}]),
        TargetRule("eating small amounts", "EATING_NO", pattern=[{"LOWER": "eating"}, {"LOWER": "small"}, {"LOWER": "amounts"}]),
        TargetRule("not eating his usual", "EATING_NO", pattern=[{"LOWER": "not"}, {"LOWER": {"IN": ["eating", "his"]}}, {"LOWER": {"IN": ["his", "her", "usual"]}}]),
    ]

    # 2. DIET BRANDS
    brand_patterns = [
        TargetRule("lyka", "DIET_BRAND"),
        TargetRule("royal canin", "DIET_BRAND", pattern=[{"LOWER": "royal"}, {"LOWER": "canin"}]),
        TargetRule("hills", "DIET_BRAND"),
        TargetRule("science diet", "DIET_BRAND", pattern=[{"LOWER": "science"}, {"LOWER": "diet"}]),
        TargetRule("blackjack", "DIET_BRAND"),
        TargetRule("black hawk", "DIET_BRAND", pattern=[{"LOWER": "black"}, {"LOWER": "hawk"}]),
        TargetRule("blackhawke", "DIET_BRAND"),
        TargetRule("nosh", "DIET_BRAND"),
        TargetRule("ziwi", "DIET_BRAND"),
        TargetRule("advance", "DIET_BRAND"),
        TargetRule("supercoat", "DIET_BRAND"),
        TargetRule("super coat", "DIET_BRAND", pattern=[{"LOWER": "super"}, {"LOWER": "coat"}]),
        TargetRule("purina", "DIET_BRAND"),
        TargetRule("proplan", "DIET_BRAND"),
        TargetRule("pro plan", "DIET_BRAND", pattern=[{"LOWER": "pro"}, {"LOWER": "plan"}]),
        TargetRule("pedigree", "DIET_BRAND"),
        TargetRule("eukanuba", "DIET_BRAND"),
        TargetRule("prime", "DIET_BRAND"),
        TargetRule("prime100", "DIET_BRAND"),
        TargetRule("prime 100", "DIET_BRAND", pattern=[{"LOWER": "prime"}, {"LOWER": "100"}]),
        TargetRule("wild pacific", "DIET_BRAND", pattern=[{"LOWER": "wild"}, {"LOWER": "pacific"}]),
        TargetRule("optimum", "DIET_BRAND"),
        TargetRule("james wellbeloved", "DIET_BRAND", pattern=[{"LOWER": "james"}, {"LOWER": "wellbeloved"}]),
        TargetRule("iams", "DIET_BRAND"),
        TargetRule("harrington", "DIET_BRAND"),
        TargetRule("arden grange", "DIET_BRAND", pattern=[{"LOWER": "arden"}, {"LOWER": "grange"}]),
        TargetRule("canagan", "DIET_BRAND"),
        TargetRule("ivory coat", "DIET_BRAND", pattern=[{"LOWER": "ivory"}, {"LOWER": "coat"}]),
        TargetRule("ivorycoat", "DIET_BRAND"),
        TargetRule("savourlife", "DIET_BRAND"),
        TargetRule("savour life", "DIET_BRAND", pattern=[{"LOWER": "savour"}, {"LOWER": "life"}]),
        TargetRule("anallergenic", "DIET_BRAND"),
        TargetRule("hypoallergenic", "DIET_BRAND"),
        TargetRule("hills prescription", "DIET_BRAND", pattern=[{"LOWER": "hills"}, {"LOWER": "prescription"}]),
        TargetRule("hills science", "DIET_BRAND", pattern=[{"LOWER": "hills"}, {"LOWER": "science"}]),
    ]

    # 3. DIET TYPES
    diet_type_patterns = [
        TargetRule("wet food", "DIET_TYPE", pattern=[{"LOWER": "wet"}, {"LOWER": "food"}]),
        TargetRule("dry food", "DIET_TYPE", pattern=[{"LOWER": "dry"}, {"LOWER": "food"}]),
        TargetRule("dry diet", "DIET_TYPE", pattern=[{"LOWER": "dry"}, {"LOWER": "diet"}]),
        TargetRule("wet diet", "DIET_TYPE", pattern=[{"LOWER": "wet"}, {"LOWER": "diet"}]),
        TargetRule("premium dry diet", "DIET_TYPE", pattern=[{"LOWER": "premium"}, {"LOWER": "dry"}, {"LOWER": "diet"}]),
        TargetRule("premium dry food", "DIET_TYPE", pattern=[{"LOWER": "premium"}, {"LOWER": "dry"}, {"LOWER": "food"}]),
        TargetRule("kibble", "DIET_TYPE"),
        TargetRule("kibbles", "DIET_TYPE"),
        TargetRule("kibbled", "DIET_TYPE"),
        TargetRule("puppy kibble", "DIET_TYPE", pattern=[{"LOWER": "puppy"}, {"LOWER": {"IN": ["kibble", "kibbles"]}}]),
        TargetRule("dry kibble", "DIET_TYPE", pattern=[{"LOWER": "dry"}, {"LOWER": "kibble"}]),
        TargetRule("biscuits", "DIET_TYPE"),
        TargetRule("tinned", "DIET_TYPE"),
        TargetRule("tinned food", "DIET_TYPE", pattern=[{"LOWER": "tinned"}, {"LOWER": "food"}]),
        TargetRule("canned food", "DIET_TYPE", pattern=[{"LOWER": "canned"}, {"LOWER": "food"}]),
        TargetRule("pouch", "DIET_TYPE"),
        TargetRule("sachet", "DIET_TYPE"),
        TargetRule("treats", "DIET_TYPE"),
        TargetRule("raw", "DIET_TYPE"),
        TargetRule("raw meat", "DIET_TYPE", pattern=[{"LOWER": "raw"}, {"LOWER": "meat"}]),
        TargetRule("raw food", "DIET_TYPE", pattern=[{"LOWER": "raw"}, {"LOWER": "food"}]),
        TargetRule("barf", "DIET_TYPE"),
        TargetRule("cooked", "DIET_TYPE"),
        TargetRule("boiled", "DIET_TYPE"),
        TargetRule("home cooked", "DIET_TYPE", pattern=[{"LOWER": "home"}, {"LOWER": "cooked"}]),
        TargetRule("cooked food", "DIET_TYPE", pattern=[{"LOWER": "cooked"}, {"LOWER": "food"}]),
        TargetRule("bland diet", "DIET_TYPE", pattern=[{"LOWER": "bland"}, {"LOWER": "diet"}]),
        TargetRule("puppy food", "DIET_TYPE", pattern=[{"LOWER": "puppy"}, {"LOWER": {"IN": ["food", "foods"]}}]),
        TargetRule("puppy formula", "DIET_TYPE", pattern=[{"LOWER": "puppy"}, {"LOWER": "formula"}]),
        TargetRule("puppy diet", "DIET_TYPE", pattern=[{"LOWER": "puppy"}, {"LOWER": "diet"}]),
        TargetRule("senior food", "DIET_TYPE", pattern=[{"LOWER": "senior"}, {"LOWER": "food"}]),
        TargetRule("senior diet", "DIET_TYPE", pattern=[{"LOWER": "senior"}, {"LOWER": "diet"}]),
        TargetRule("adult food", "DIET_TYPE", pattern=[{"LOWER": "adult"}, {"LOWER": "food"}]),
        TargetRule("adult diet", "DIET_TYPE", pattern=[{"LOWER": "adult"}, {"LOWER": "diet"}]),
        TargetRule("dog food", "DIET_TYPE", pattern=[{"LOWER": "dog"}, {"LOWER": "food"}]),
        TargetRule("cat food", "DIET_TYPE", pattern=[{"LOWER": "cat"}, {"LOWER": "food"}]),
        TargetRule("dog roll", "DIET_TYPE", pattern=[{"LOWER": "dog"}, {"LOWER": "roll"}]),
        TargetRule("table scraps", "DIET_TYPE", pattern=[{"LOWER": "table"}, {"LOWER": "scraps"}]),
        TargetRule("scraps", "DIET_TYPE"),
        TargetRule("bones", "DIET_TYPE"),
        TargetRule("pig ears", "DIET_TYPE", pattern=[{"LOWER": "pig"}, {"LOWER": "ears"}]),
        TargetRule("dental diet", "DIET_TYPE", pattern=[{"LOWER": "dental"}, {"LOWER": "diet"}]),
        TargetRule("prescription diet", "DIET_TYPE", pattern=[{"LOWER": "prescription"}, {"LOWER": "diet"}]),
        TargetRule("elimination diet", "DIET_TYPE", pattern=[{"LOWER": "elimination"}, {"LOWER": "diet"}]),
        TargetRule("low fat diet", "DIET_TYPE", pattern=[{"LOWER": "low"}, {"LOWER": "fat"}, {"LOWER": "diet"}]),
        TargetRule("fresh food", "DIET_TYPE", pattern=[{"LOWER": "fresh"}, {"LOWER": "food"}]),
        TargetRule("fresh food diet", "DIET_TYPE", pattern=[{"LOWER": "fresh"}, {"LOWER": "food"}, {"LOWER": "diet"}]),
        TargetRule("chicken", "DIET_ITEM"),
        TargetRule("boiled chicken", "DIET_ITEM", pattern=[{"LOWER": "boiled"}, {"LOWER": "chicken"}]),
        TargetRule("chicken breast", "DIET_ITEM", pattern=[{"LOWER": "chicken"}, {"LOWER": "breast"}]),
        TargetRule("cooked chicken", "DIET_ITEM", pattern=[{"LOWER": "cooked"}, {"LOWER": "chicken"}]),
        TargetRule("beef", "DIET_ITEM"),
        TargetRule("lamb", "DIET_ITEM"),
        TargetRule("turkey", "DIET_ITEM"),
        TargetRule("kangaroo", "DIET_ITEM"),
        TargetRule("fish", "DIET_ITEM"),
        TargetRule("salmon", "DIET_ITEM"),
        TargetRule("tuna", "DIET_ITEM"),
        TargetRule("sardines", "DIET_ITEM"),
        TargetRule("duck", "DIET_ITEM"),
        TargetRule("venison", "DIET_ITEM"),
        TargetRule("rabbit", "DIET_ITEM"),
        TargetRule("pork", "DIET_ITEM"),
        TargetRule("mince", "DIET_ITEM"),
        TargetRule("chicken mince", "DIET_ITEM", pattern=[{"LOWER": "chicken"}, {"LOWER": "mince"}]),
        TargetRule("rice", "DIET_ITEM"),
        TargetRule("white rice", "DIET_ITEM", pattern=[{"LOWER": "white"}, {"LOWER": "rice"}]),
        TargetRule("cooked rice", "DIET_ITEM", pattern=[{"LOWER": "cooked"}, {"LOWER": "rice"}]),
        TargetRule("pasta", "DIET_ITEM"),
        TargetRule("potato", "DIET_ITEM"),
        TargetRule("sweet potato", "DIET_ITEM", pattern=[{"LOWER": "sweet"}, {"LOWER": "potato"}]),
        TargetRule("pumpkin", "DIET_ITEM"),
        TargetRule("vegetables", "DIET_ITEM"),
        TargetRule("greens", "DIET_ITEM"),
        TargetRule("carrot", "DIET_ITEM"),
        TargetRule("blueberries", "DIET_ITEM"),
    ]

    # 4. EATING BEHAVIORS
    behavior_patterns = [
        TargetRule("vomiting", "BEHAVIOR_GI"),
        TargetRule("vomits", "BEHAVIOR_GI"),
        TargetRule("vomit", "BEHAVIOR_GI"),
        TargetRule("diarrhoea", "BEHAVIOR_GI"),
        TargetRule("diarrhea", "BEHAVIOR_GI"),
        TargetRule("loose stool", "BEHAVIOR_GI", pattern=[{"LOWER": "loose"}, {"LOWER": {"IN": ["stool", "stools"]}}]),
        TargetRule("soft stool", "BEHAVIOR_GI", pattern=[{"LOWER": "soft"}, {"LOWER": {"IN": ["stool", "stools"]}}]),
        TargetRule("bloody diarrhoea", "BEHAVIOR_GI", pattern=[{"LOWER": "bloody"}, {"LOWER": {"IN": ["diarrhoea", "diarrhea", "faeces"]}}]),
        TargetRule("blood in faeces", "BEHAVIOR_GI", pattern=[{"LOWER": "blood"}, {"LOWER": "in"}, {"LOWER": {"IN": ["faeces", "feces", "stool", "stools"]}}]),
        TargetRule("bloody faeces", "BEHAVIOR_GI", pattern=[{"LOWER": "bloody"}, {"LOWER": {"IN": ["faeces", "feces"]}}]),
        TargetRule("blood", "BEHAVIOR_GI"),
        TargetRule("constipation", "BEHAVIOR_GI"),
        TargetRule("constipated", "BEHAVIOR_GI"),
        TargetRule("nausea", "BEHAVIOR_GI"),
        TargetRule("retching", "BEHAVIOR_GI"),
        TargetRule("flatulence", "BEHAVIOR_GI"),
        TargetRule("bloating", "BEHAVIOR_GI"),
        TargetRule("regurgitation", "BEHAVIOR_GI"),
        TargetRule("no vomiting or diarrhoea", "BEHAVIOR_NORMAL", pattern=[{"LOWER": "no"}, {"LOWER": "vomiting"}, {"LOWER": "or"}, {"LOWER": "diarrhoea"}]),
        TargetRule("no vomiting diarrhoea", "BEHAVIOR_NORMAL", pattern=[{"LOWER": "no"}, {"LOWER": "vomiting"}, {"LOWER": "diarrhoea"}]),
        TargetRule("no diarrhoea", "BEHAVIOR_NORMAL", pattern=[{"LOWER": "no"}, {"LOWER": {"IN": ["diarrhoea", "diarrhea"]}}]),
        TargetRule("no vomiting", "BEHAVIOR_NORMAL", pattern=[{"LOWER": "no"}, {"LOWER": "vomiting"}]),
        TargetRule("lethargic", "BEHAVIOR_GENERAL"),
        TargetRule("lethargy", "BEHAVIOR_GENERAL"),
        TargetRule("weakness", "BEHAVIOR_GENERAL"),
        TargetRule("tired", "BEHAVIOR_GENERAL"),
        TargetRule("fussy", "BEHAVIOR_EATING"),
        TargetRule("fussy eater", "BEHAVIOR_EATING", pattern=[{"LOWER": "fussy"}, {"LOWER": "eater"}]),
        TargetRule("grazing", "BEHAVIOR_EATING"),
        TargetRule("refusing", "BEHAVIOR_EATING"),
        TargetRule("picky", "BEHAVIOR_EATING"),
        TargetRule("picky eater", "BEHAVIOR_EATING", pattern=[{"LOWER": "picky"}, {"LOWER": "eater"}]),
        TargetRule("coprophagia", "BEHAVIOR_EATING"),
        TargetRule("weight loss", "BEHAVIOR_WEIGHT", pattern=[{"LOWER": "weight"}, {"LOWER": "loss"}]),
        TargetRule("losing weight", "BEHAVIOR_WEIGHT", pattern=[{"LOWER": "losing"}, {"LOWER": "weight"}]),
        TargetRule("lost weight", "BEHAVIOR_WEIGHT", pattern=[{"LOWER": "lost"}, {"LOWER": {"IN": ["weight", "some"]}}]),
        TargetRule("weight gain", "BEHAVIOR_WEIGHT", pattern=[{"LOWER": "weight"}, {"LOWER": "gain"}]),
        TargetRule("gaining weight", "BEHAVIOR_WEIGHT", pattern=[{"LOWER": "gaining"}, {"LOWER": "weight"}]),
        TargetRule("not gaining weight", "BEHAVIOR_WEIGHT", pattern=[{"LOWER": "not"}, {"LOWER": "gaining"}, {"LOWER": "weight"}]),
        TargetRule("thirst", "BEHAVIOR_GENERAL"),
        TargetRule("increased thirst", "BEHAVIOR_GENERAL", pattern=[{"LOWER": "increased"}, {"LOWER": "thirst"}]),
        TargetRule("polydipsia", "BEHAVIOR_GENERAL"),
    ]

    # 5. DIET CHANGES
    diet_change_patterns = [
        TargetRule("changed to", "DIET_CHANGE", pattern=[{"LOWER": "changed"}, {"LOWER": "to"}]),
        TargetRule("changing to", "DIET_CHANGE", pattern=[{"LOWER": "changing"}, {"LOWER": "to"}]),
        TargetRule("switched to", "DIET_CHANGE", pattern=[{"LOWER": "switched"}, {"LOWER": "to"}]),
        TargetRule("switching to", "DIET_CHANGE", pattern=[{"LOWER": "switching"}, {"LOWER": "to"}]),
        TargetRule("transition to", "DIET_CHANGE", pattern=[{"LOWER": {"IN": ["transition", "transitioning"]}}, {"LOWER": "to"}]),
        TargetRule("started on", "DIET_CHANGE", pattern=[{"LOWER": "started"}, {"LOWER": "on"}]),
        TargetRule("starting on", "DIET_CHANGE", pattern=[{"LOWER": "starting"}, {"LOWER": "on"}]),
        TargetRule("moved to", "DIET_CHANGE", pattern=[{"LOWER": "moved"}, {"LOWER": "to"}]),
        TargetRule("moving to", "DIET_CHANGE", pattern=[{"LOWER": "moving"}, {"LOWER": "to"}]),
        TargetRule("trialing", "DIET_CHANGE"),
        TargetRule("trial", "DIET_CHANGE"),
        TargetRule("trying", "DIET_CHANGE"),
        TargetRule("discussed diet", "DIET_DISCUSSED", pattern=[{"LOWER": "discussed"}, {"LOWER": {"IN": ["diet", "food", "nutrition"]}}]),
        TargetRule("discuss diet", "DIET_DISCUSSED", pattern=[{"LOWER": "discuss"}, {"LOWER": {"IN": ["diet", "food", "nutrition"]}}]),
        TargetRule("diet discussed", "DIET_DISCUSSED", pattern=[{"LOWER": "diet"}, {"LOWER": "discussed"}]),
        TargetRule("food discussed", "DIET_DISCUSSED", pattern=[{"LOWER": "food"}, {"LOWER": "discussed"}]),
        TargetRule("discussed food", "DIET_DISCUSSED", pattern=[{"LOWER": "discussed"}, {"LOWER": "food"}]),
        TargetRule("recommended", "DIET_DISCUSSED"),
        TargetRule("suggest", "DIET_DISCUSSED"),
        TargetRule("suggested", "DIET_DISCUSSED"),
        TargetRule("plan to change", "DIET_PLANNED", pattern=[{"LOWER": "plan"}, {"LOWER": "to"}, {"LOWER": "change"}]),
        TargetRule("planning to change", "DIET_PLANNED", pattern=[{"LOWER": "planning"}, {"LOWER": "to"}, {"LOWER": "change"}]),
        TargetRule("will change", "DIET_PLANNED", pattern=[{"LOWER": "will"}, {"LOWER": "change"}]),
        TargetRule("going to change", "DIET_PLANNED", pattern=[{"LOWER": "going"}, {"LOWER": "to"}, {"LOWER": "change"}]),
        TargetRule("no change", "DIET_NO_CHANGE", pattern=[{"LOWER": "no"}, {"LOWER": "change"}]),
        TargetRule("continue", "DIET_NO_CHANGE"),
        TargetRule("continuing", "DIET_NO_CHANGE"),
        TargetRule("same diet", "DIET_NO_CHANGE", pattern=[{"LOWER": "same"}, {"LOWER": {"IN": ["diet", "food"]}}]),
        TargetRule("same food", "DIET_NO_CHANGE", pattern=[{"LOWER": "same"}, {"LOWER": "food"}]),
    ]

    return (
        eating_yes_patterns
        + eating_no_patterns
        + brand_patterns
        + diet_type_patterns
        + behavior_patterns
        + diet_change_patterns
    )


def build_context_rules() -> List[ConTextRule]:
    negation_rules = [
        ConTextRule("no", "NEGATED_EXISTENCE", direction="forward", max_scope=3),
        ConTextRule("not", "NEGATED_EXISTENCE", direction="forward", max_scope=2),
        ConTextRule("never", "NEGATED_EXISTENCE", direction="forward", max_scope=3),
        ConTextRule("without", "NEGATED_EXISTENCE", direction="forward", max_scope=3),
        ConTextRule("none", "NEGATED_EXISTENCE", direction="forward", max_scope=2),
        ConTextRule("denies", "NEGATED_EXISTENCE", direction="forward", max_scope=3),
        ConTextRule("denied", "NEGATED_EXISTENCE", direction="forward", max_scope=3),
        ConTextRule("refuses", "NEGATED_EXISTENCE", direction="forward", max_scope=2),
        ConTextRule("refused", "NEGATED_EXISTENCE", direction="forward", max_scope=2),
        ConTextRule("absence of", "NEGATED_EXISTENCE", direction="forward", max_scope=3),
        ConTextRule("free of", "NEGATED_EXISTENCE", direction="forward", max_scope=3),
        ConTextRule("negative for", "NEGATED_EXISTENCE", direction="forward", max_scope=3),
        # Patient name context - negates diet items that are actually patient names
        ConTextRule("patient name", "NEGATED_EXISTENCE", direction="forward", max_scope=2,
                    allowed_types={"DIET_ITEM", "DIET_BRAND"}),
        ConTextRule("name", "NEGATED_EXISTENCE", direction="forward", max_scope=1,
                    allowed_types={"DIET_ITEM", "DIET_BRAND"}),
        ConTextRule("called", "NEGATED_EXISTENCE", direction="forward", max_scope=1,
                    allowed_types={"DIET_ITEM", "DIET_BRAND"}),
        ConTextRule("named", "NEGATED_EXISTENCE", direction="forward", max_scope=1,
                    allowed_types={"DIET_ITEM", "DIET_BRAND"}),
    ]

    temporality_rules = [
        ConTextRule("history of", "HISTORICAL", direction="forward", max_scope=4),
        ConTextRule("history", "HISTORICAL", direction="forward", max_scope=1),
        ConTextRule("previous", "HISTORICAL", direction="forward", max_scope=2),
        ConTextRule("previously", "HISTORICAL", direction="forward", max_scope=3),
        ConTextRule("past", "HISTORICAL", direction="forward", max_scope=2),
        ConTextRule("chronic", "HISTORICAL", direction="forward", max_scope=2),
        ConTextRule("since", "HISTORICAL", direction="backward", max_scope=3),
        ConTextRule("ongoing", "HISTORICAL", direction="forward", max_scope=2),
        ConTextRule("recent", "HISTORICAL", direction="forward", max_scope=2),
        ConTextRule("last week", "HISTORICAL", direction="bidirectional", max_scope=3),
        ConTextRule("last month", "HISTORICAL", direction="bidirectional", max_scope=3),
        ConTextRule("yesterday", "HISTORICAL", direction="bidirectional", max_scope=2),
        ConTextRule("days ago", "HISTORICAL", direction="backward", max_scope=3),
        ConTextRule("weeks ago", "HISTORICAL", direction="backward", max_scope=3),
        ConTextRule("months ago", "HISTORICAL", direction="backward", max_scope=3),
        ConTextRule("ago", "HISTORICAL", direction="backward", max_scope=3),
        ConTextRule("was", "HISTORICAL", direction="forward", max_scope=1),
        ConTextRule("were", "HISTORICAL", direction="forward", max_scope=1),
        ConTextRule("had been", "HISTORICAL", direction="forward", max_scope=3),
        ConTextRule("had", "HISTORICAL", direction="forward", max_scope=2),
    ]

    hypothetical_rules = [
        ConTextRule("if", "HYPOTHETICAL", direction="forward", max_scope=5),
        ConTextRule("may", "HYPOTHETICAL", direction="forward", max_scope=2),
        ConTextRule("might", "HYPOTHETICAL", direction="forward", max_scope=2),
        ConTextRule("could", "HYPOTHETICAL", direction="forward", max_scope=2),
        ConTextRule("would", "HYPOTHETICAL", direction="forward", max_scope=2),
        ConTextRule("will", "HYPOTHETICAL", direction="forward", max_scope=3),
        ConTextRule("recommend", "HYPOTHETICAL", direction="forward", max_scope=2),
        ConTextRule("recommended", "HYPOTHETICAL", direction="forward", max_scope=2),
        ConTextRule("advise", "HYPOTHETICAL", direction="forward", max_scope=2),
        ConTextRule("advised", "HYPOTHETICAL", direction="forward", max_scope=2),
        ConTextRule("suggest", "HYPOTHETICAL", direction="forward", max_scope=2),
        ConTextRule("suggested", "HYPOTHETICAL", direction="forward", max_scope=2),
        ConTextRule("trial", "HYPOTHETICAL", direction="bidirectional", max_scope=2),
        ConTextRule("consider", "HYPOTHETICAL", direction="forward", max_scope=3),
        ConTextRule("considering", "HYPOTHETICAL", direction="forward", max_scope=3),
        ConTextRule("plan to", "HYPOTHETICAL", direction="forward", max_scope=4),
        ConTextRule("planning", "HYPOTHETICAL", direction="forward", max_scope=3),
        ConTextRule("possible", "HYPOTHETICAL", direction="bidirectional", max_scope=2),
        ConTextRule("potential", "HYPOTHETICAL", direction="forward", max_scope=2),
    ]

    return negation_rules + temporality_rules + hypothetical_rules


def build_pipeline() -> spacy.language.Language:
    nlp = medspacy.load()
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add(build_target_rules())
    context = nlp.get_pipe("medspacy_context")
    context.add(build_context_rules())
    return nlp


def extract_eating_status(nlp: spacy.language.Language, text: str) -> str:
    if pd.isna(text) or str(text).strip() == "":
        return "Unknown"

    text_to_process = str(text)
    pet_name = detect_pet_name(text_to_process)
    if pet_name:
        text_to_process = mask_pet_name(text_to_process, pet_name)

    doc = nlp(text_to_process)

    eating_yes_count = 0
    eating_no_count = 0

    for ent in doc.ents:
        is_negated = ent._.is_negated
        is_historical = ent._.is_historical
        is_hypothetical = ent._.is_hypothetical

        if not is_historical and not is_hypothetical:
            if ent.label_ == "EATING_YES" and not is_negated:
                eating_yes_count += 1
            elif ent.label_ == "EATING_NO" and not is_negated:
                eating_no_count += 1
            elif ent.label_ == "BEHAVIOR_NORMAL":
                eating_yes_count += 1

            if ent.label_ == "EATING_NO" and is_negated:
                eating_yes_count += 1

    if eating_no_count > eating_yes_count:
        return "No"
    if eating_yes_count > eating_no_count:
        return "Yes"
    return "Unknown"


def check_nutrition_info_present(nlp: spacy.language.Language, text: str) -> str:
    if pd.isna(text) or str(text).strip() == "":
        return "No"

    text_to_process = str(text)
    pet_name = detect_pet_name(text_to_process)
    if pet_name:
        text_to_process = mask_pet_name(text_to_process, pet_name)

    doc = nlp(text_to_process)

    # Exclude entities that are negated (e.g., patient names) or hypothetical
    diet_entities = [
        ent
        for ent in doc.ents
        if ent.label_ in {"DIET_BRAND", "DIET_ITEM", "DIET_TYPE"}
        and not ent._.is_hypothetical
        and not ent._.is_negated  # Exclude patient names (negated by "patient name" context)
    ]

    return "Yes" if diet_entities else "No"


def check_nutrition_info_present_batch(nlp: spacy.language.Language, texts: List[str]) -> List[str]:
    results: List[str] = []
    with nlp.select_pipes(disable=["medspacy_context"]):
        for raw_text in texts:
            if pd.isna(raw_text) or str(raw_text).strip() == "":
                results.append("No")
                continue

            text_to_process = str(raw_text)
            pet_name = detect_pet_name(text_to_process)
            if pet_name:
                text_to_process = mask_pet_name(text_to_process, pet_name)

            found = False
            for ent in nlp(text_to_process).ents:
                if ent.label_ in {"DIET_BRAND", "DIET_ITEM", "DIET_TYPE"}:
                    found = True
                    break
            results.append("Yes" if found else "No")
    return results


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if ANNOTATION_COLUMN in df.columns:
        df = df[df[ANNOTATION_COLUMN].notna()].copy()
    return df


def evaluate_against_annotation(df: pd.DataFrame) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        df[ANNOTATION_COLUMN], df["nutrition_info_present"], average="weighted"
    )

    tn, fp, fn, tp = confusion_matrix(df[ANNOTATION_COLUMN], df["nutrition_info_present"]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(df[ANNOTATION_COLUMN], df["nutrition_info_present"])

    print("\n" + "=" * 70)
    print("DETAILED METRICS")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print("=" * 70)
    print("CLINICALLY RELEVANT METRICS")
    print("=" * 70)
    print(f"Accuracy:     {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Sensitivity:  {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
    print(f"Specificity:  {specificity:.4f} ({specificity * 100:.2f}%)")
    print("=" * 70)


def plot_confusion_matrix(df: pd.DataFrame) -> None:
    """Plot confusion matrix for the classification results."""
    cm = confusion_matrix(df[ANNOTATION_COLUMN], df["nutrition_info_present"])
    annotations = [[cm[i, j] for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    sns.heatmap(
        cm,
        annot=np.array(annotations),
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        linewidths=2,
        linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
    )
    import matplotlib.pyplot as plt
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {DATA_PATH}")

    nlp = build_pipeline()
    df = load_dataframe(DATA_PATH)

    # df["patient_eating"] = df["cleaned_examination_text"].apply(lambda t: extract_eating_status(nlp, t))
    df["nutrition_info_present"] = df["cleaned_examination_text"].apply(
        lambda t: check_nutrition_info_present(nlp, t)
    )

    df.to_csv(OUTPUT_DIR / "df_rulesClassification.csv", index=False)
    df.to_excel(OUTPUT_DIR / "df_rulesClassification.xlsx", index=False)

    evaluate_against_annotation(df)
    plot_confusion_matrix(df)

    # Identify false positives/negatives
    if ANNOTATION_COLUMN in df.columns:
        false_positives = df[(df[ANNOTATION_COLUMN] == "No") & (df["nutrition_info_present"] == "Yes")].copy().reset_index()
        false_negatives = df[(df[ANNOTATION_COLUMN] == "Yes") & (df["nutrition_info_present"] == "No")].copy().reset_index()
        false_positives.to_csv(OUTPUT_DIR / "false_positives.csv", index=False)
        false_negatives.to_csv(OUTPUT_DIR / "false_negatives.csv", index=False)
        print(f"\nFalse positives: {len(false_positives)}")
        print(f"False negatives: {len(false_negatives)}")


if __name__ == "__main__":
    main()



# ======================================================================
# DETAILED METRICS
# ======================================================================
# Accuracy:  0.8273 (82.73%)
# Precision: 0.8299 (82.99%)
# Recall:    0.8273 (82.73%)
# F1-Score:  0.8249 (82.49%)
# ======================================================================
# CLINICALLY RELEVANT METRICS
# ======================================================================
# Accuracy:     0.8273 (82.73%)
# Sensitivity:  0.7233 (72.33%)
# Specificity:  0.9056 (90.56%)
# ======================================================================



# ======================================================================
# DETAILED METRICS (After optimization of context rules)
# ======================================================================
# Accuracy:  0.7196 (71.96%)
# Precision: 0.7614 (76.14%)
# Recall:    0.7196 (71.96%)
# F1-Score:  0.6925 (69.25%)
# ======================================================================
# CLINICALLY RELEVANT METRICS
# ======================================================================
# Accuracy:     0.7196 (71.96%)
# Sensitivity:  0.4093 (40.93%)
# Specificity:  0.9528 (95.28%)
# ======================================================================


# ======================================================================
# DETAILED METRICS (After adding DIET_TYPE)
# ======================================================================
# Accuracy:  0.7545 (75.45%)
# Precision: 0.7741 (77.41%)
# Recall:    0.7545 (75.45%)
# F1-Score:  0.7412 (74.12%)
# ======================================================================
# CLINICALLY RELEVANT METRICS
# ======================================================================
# Accuracy:     0.7545 (75.45%)
# Sensitivity:  0.5256 (52.56%)
# Specificity:  0.9266 (92.66%)
# ======================================================================