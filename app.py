import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.pipline.prediction_pipeline import PredictionPipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os

app = FastAPI(title="User Segmentation App")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DATA_PATH = "data/data_with_clusters.csv"
CLUSTER_LABELS = {
    0: "Weekend Warriors",
    1: "Engaged Professionals",
    2: "Low-Key Users",
    3: "Active Explorers",
    4: "Budget Browsers"
}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not os.path.exists(DATA_PATH):
        return HTMLResponse("<h2>Data file not found.</h2>", status_code=404)

    df = pd.read_csv(DATA_PATH)
    df["Cluster Label"] = df["cluster"].map(CLUSTER_LABELS)

    total_users = len(df)
    avg_ctr = round(df['Click-Through Rates (CTR)'].mean(), 2)
    avg_weekday = round(df['Time Spent Online (hrs/weekday)'].mean(), 2)
    avg_conversion = round(df['Conversion Rates'].mean(), 2)

    cluster_data = df["Cluster Label"].value_counts().reset_index()
    cluster_data.columns = ["Segment", "Users"]
    bar_fig = px.bar(cluster_data, x="Segment", y="Users", text="Users", title="Users per Segment")
    bar_html = pio.to_html(bar_fig, full_html=False)

    features = ["Time Spent Online (hrs/weekday)", "Time Spent Online (hrs/weekend)", "Likes and Reactions", "Click-Through Rates (CTR)"]
    radar_df = df.groupby("Cluster Label")[features].mean()
    radar_df_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())
    radar_df_norm = radar_df_norm.reset_index()
    radar_fig = go.Figure()
    for i, label in enumerate(radar_df_norm["Cluster Label"]):
        vals = radar_df_norm.iloc[i][features].tolist()
        radar_fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=features + [features[0]], fill='toself', name=label))
    radar_fig.update_layout(title="Segment Profiles", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    radar_html = pio.to_html(radar_fig, full_html=False)

    gender_fig = px.histogram(df, x="Gender", color="Cluster Label", barmode="group", title="Gender by Segment")
    gender_html = pio.to_html(gender_fig, full_html=False)

    income_fig = px.histogram(df, x="Income Level", color="Cluster Label", barmode="group", title="Income vs Segment")
    income_html = pio.to_html(income_fig, full_html=False)

    top_interests = df["Top Interests"].dropna().str.split(", ").explode().value_counts().nlargest(10)
    interest_fig = px.bar(top_interests, x=top_interests.index, y=top_interests.values,
                          labels={"x": "Interest", "y": "Users"}, title="Top 10 Interests")
    interest_html = pio.to_html(interest_fig, full_html=False)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total_users": total_users,
        "avg_ctr": avg_ctr,
        "avg_weekday": avg_weekday,
        "avg_conversion": avg_conversion,
        "bar_html": bar_html,
        "radar_html": radar_html,
        "gender_html": gender_html,
        "income_html": income_html,
        "interest_html": interest_html
    })

@app.get("/predict", response_class=HTMLResponse)
async def show_predict_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_cluster(request: Request,
    Age: str = Form(...),
    Gender: str = Form(...),
    Income_Level: str = Form(...),
    Time_Spent_Online_hrs_weekday: float = Form(...),
    Time_Spent_Online_hrs_weekend: float = Form(...),
    Likes_and_Reactions: int = Form(...),
    Click_Through_Rates_CTR: float = Form(...)):
    try:
        input_data = {
            "Time Spent Online (hrs/weekday)": Time_Spent_Online_hrs_weekday,
            "Time Spent Online (hrs/weekend)": Time_Spent_Online_hrs_weekend,
            "Likes and Reactions": Likes_and_Reactions,
            "Click-Through Rates (CTR)": Click_Through_Rates_CTR,
            "Age": Age,
            "Gender": Gender,
            "Income Level": Income_Level
        }
        pipeline = PredictionPipeline()
        cluster = pipeline.predict(input_data)
        label = CLUSTER_LABELS.get(cluster, "Unknown")

        return templates.TemplateResponse("results.html", {
            "request": request,
            "cluster": cluster,
            "label": label
        })

    except Exception as e:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "cluster": "Error",
            "label": str(e)
        })
    
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)




