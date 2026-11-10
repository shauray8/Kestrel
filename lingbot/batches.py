import json, os, sys
S, C = "/workspace/scenes", "/workspace/cams"
HERO = ["desert_highway", "neon_street", "mountain_pine_road", "coastal_road", "snowy_forest_road", "signage_street"]
ACC = ["neon_street", "mountain_pine_road", "signage_street"]


def item(scene, script, frames, tag):
    return {"name": f"{tag}/{scene}_{script}_f{frames}", "image": f"{S}/{scene}.png", "prompt": f"{S}/{scene}.txt", "action": f"{C}/{script}", "frames": frames}


B = {
    "hero": [item(s, "A_push", 97, "hero") for s in HERO],
    "acc": [item(s, "C_lookback", 361, "accuracy") for s in ACC],
    "extra": [item("desert_highway", "B_arc", 97, "diagram"), item("signage_street", "D_strafe", 97, "strafe"), item("mountain_pine_road", "D_strafe", 97, "strafe")],
}
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for k, v in B.items():
    json.dump(v, open(f"{out}/{k}.json", "w"), indent=1)
for combo in (("hero", "acc", "extra"), ("acc", "extra")):
    json.dump(sum((B[k] for k in combo), []), open(f"{out}/{'_'.join(combo)}.json", "w"), indent=1)
print(os.listdir(out))
