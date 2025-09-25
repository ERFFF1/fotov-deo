from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/")
def root():
    return {"ok": True, "service": "face-video-ai-studio"}

@app.get("/dashboard")
def dashboard(req: Request):
    return templates.TemplateResponse("dashboard.html", {"request": req})