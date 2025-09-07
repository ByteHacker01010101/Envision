from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

app = Flask(__name__)
CORS(app)

DF_PATH = "datasets/cosmetics.csv"
df = pd.read_csv(DF_PATH)

if "Name" not in df.columns:
    df.rename(columns={df.columns[0]: "Name"}, inplace=True)
for col in ["Ingredients", "Label", "Brand"]:
    if col not in df.columns:
        df[col] = ""
df["Ingredients"] = df["Ingredients"].fillna("")
df = df.reset_index().rename(columns={"index": "id"})

# Build ingredient vocab
all_ing = set()
for s in df["Ingredients"]:
    toks = [t.strip().lower() for t in str(s).split(",") if t.strip()]
    all_ing.update(toks)
all_ing = sorted(all_ing)
ingredient_idx = {ing: i for i, ing in enumerate(all_ing)}

def encode_ingredients(text):
    if len(ingredient_idx) == 0:
        return np.zeros(0)
    vec = np.zeros(len(ingredient_idx), dtype=np.uint8)
    for ing in [t.strip().lower() for t in str(text).split(",") if t.strip()]:
        if ing in ingredient_idx:
            vec[ingredient_idx[ing]] = 1
    return vec

if len(all_ing) > 0:
    ingredient_matrix = np.vstack([encode_ingredients(t) for t in df["Ingredients"]])
else:
    ingredient_matrix = np.zeros((len(df), 0))

@app.route("/products", methods=["GET"])
def get_products():
    label = request.args.get("label", type=str)
    brand = request.args.get("brand", type=str)
    q = request.args.get("q", type=str)
    skins = {s: request.args.get(s, default=None) for s in ["Combination", "Dry", "Normal", "Oily", "Sensitive"]}

    filtered = df.copy()
    if label:
        filtered = filtered[filtered["Label"] == label]
    if brand:
        filtered = filtered[filtered["Brand"] == brand]
    for s, val in skins.items():
        if val is not None:
            try:
                v = int(val)
            except:
                v = 1
            if s in filtered.columns and v == 1:
                filtered = filtered[filtered[s] == 1]
    if q:
        filtered = filtered[filtered["Name"].str.contains(q, case=False, na=False)]
    return jsonify(filtered.to_dict(orient="records"))

@app.route("/brands", methods=["GET"])
def get_brands():
    brands = sorted(df["Brand"].dropna().unique().tolist())
    return jsonify(brands)

@app.route("/product/<int:pid>", methods=["GET"])
def get_product(pid):
    rec = df[df["id"] == pid]
    if rec.empty:
        return jsonify({"error": "not found"}), 404
    return jsonify(rec.iloc[0].to_dict())

@app.route("/recommend/<int:pid>", methods=["GET"])
def recommend(pid):
    if pid < 0 or pid >= len(df):
        return jsonify({"error": "invalid id"}), 404
    if ingredient_matrix.size == 0:
        return jsonify([])
    vec = ingredient_matrix[pid].reshape(1, -1)
    sims = cosine_similarity(vec, ingredient_matrix)[0]
    top = sims.argsort()[::-1]
    top = [i for i in top if i != pid][:6]
    recs = df[df['id'].isin(top)].to_dict(orient="records")
    return jsonify(recs)

@app.route("/embedding", methods=["GET"])
def embedding():
    if ingredient_matrix.size == 0:
        return jsonify([])
    perplexity = request.args.get("perplexity", default=30, type=float)
    n_iter = request.args.get("n_iter", default=1000, type=int)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter)
    coords = tsne.fit_transform(ingredient_matrix.astype(float))
    out = []
    for i, (x, y) in enumerate(coords):
        row = df.loc[df['id'] == i].iloc[0]
        out.append({"id": int(i), "x": float(x), "y": float(y), "Name": str(row.get("Name", "")), "Label": str(row.get("Label", ""))})
    return jsonify(out)

@app.route("/predict", methods=["GET"])
def predict():
    skin = request.args.get("skin", type=str)
    goal = request.args.get("goal", type=str)
    preference = request.args.get("preference", type=str)
    goal_map = {
        'Hydration': ['Moisturizer', 'Treatment'],
        'Acne': ['Treatment', 'Cleanser'],
        'Anti-aging': ['Treatment', 'Moisturizer', 'Eye cream'],
        'Sun protection': ['Sun Protect'],
        'Brightening': ['Treatment', 'Moisturizer']
    }
    pref_keywords = {
        'natural': ['aloe', 'butter', 'oil', 'extract', 'plant', 'herb'],
        'chemical': ['retinol', 'salicylic', 'glycolic', 'niacinamide', 'benzoyl']
    }
    scores = []
    for _, row in df.iterrows():
        sc = 0
        if skin and skin in df.columns and str(row.get(skin)) == '1':
            sc += 2
        if goal:
            labels_ok = goal_map.get(goal, [])
            if str(row.get('Label')) in labels_ok:
                sc += 1
        if preference and preference in pref_keywords:
            ing = str(row.get('Ingredients','')).lower()
            for kw in pref_keywords[preference]:
                if kw in ing:
                    sc += 1
                    break
        scores.append((int(row['id']), sc))
    scored = sorted(scores, key=lambda x: x[1], reverse=True)
    top_ids = [i for i, s in scored if s>0][:5]
    if not top_ids:
        top_ids = df.sort_values('Rank', ascending=False).head(5)['id'].tolist()
    recs = df[df['id'].isin(top_ids)].to_dict(orient='records')
    return jsonify(recs)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
