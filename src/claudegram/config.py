import os
from dotenv import load_dotenv

load_dotenv()

CLAUDE_API_KEY     = os.environ["CLAUDE_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

CLAUDE_MODEL            = os.environ["CLAUDE_MODEL"]
CLAUDE_BOT_USERNAME     = os.environ["CLAUDE_BOT_USERNAME"]
CLAUDE_BOT_DISPLAY_NAME = os.environ["CLAUDE_BOT_DISPLAY_NAME"]

TOKEN_BUDGET  = int(os.environ["TOKEN_BUDGET"])
REPLY_BUDGET  = int(os.environ["REPLY_BUDGET"])
DATA_DIR      = os.environ.get("DATA_DIR", "data")
LOG_DIR       = os.environ.get("LOG_DIR", "logs")
IGNORE_PREFIX = os.environ.get("IGNORE_PREFIX")

ADMIN_USER_IDS = {
    int(x) for x in os.environ.get("ADMIN_USER_IDS", "").split(",") if x.strip()
}