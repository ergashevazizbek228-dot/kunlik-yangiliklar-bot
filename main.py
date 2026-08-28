# -*- coding: utf-8 -*-
"""
Telegram AI Yangiliklar Agenti (rasm + sifat nazorati bilan)
--------------------------------------------------------------
1. sources.py'dagi RSS manbalardan eng so'nggi yangiliklarni o'qiydi
2. Har bir yangi xabarni Groq API (bepul) orqali o'zbek tilida qayta yozadi
3. SIFAT NAZORATI: postni AI orqali tekshiradi (aniqlik, uzunlik, mos-
   lik, nomaqbul kontent yo'qligi). O'tmasa - qayta yozishga urinadi,
   baribir o'tmasa - postni o'tkazib yuboradi.
4. RASM: avval manba xabaridan yoki maqola sahifasidan (og:image) haqiqiy
   rasmni topishga harakat qiladi. Topilmasa - Google Gemini 2.5 Flash
   Image ("Nano Banana", bepul limit bilan) orqali post uchun mos rasm
   generatsiya qiladi.
5. Tayyor post (surat + matn) Telegram kanaliga joylanadi.
6. Joylangan xabarlar ro'yxatini posted_ids.json'da saqlaydi (takrorlanmasin).

Kerakli muhit o'zgaruvchilari (environment variables):
  TELEGRAM_BOT_TOKEN   - BotFather bergan token
  TELEGRAM_CHAT_ID     - kanal username (masalan @mening_kanalim) yoki chat id
  GROQ_API_KEY         - https://console.groq.com dan bepul olinadigan kalit (matn uchun)
  GEMINI_API_KEY       - https://aistudio.google.com/apikey dan bepul olinadigan kalit (rasm uchun)
"""

import os
import re
import json
import time
import base64
import hashlib
import sys

import requests
import feedparser

from sources import SOURCES, ENTRIES_PER_SOURCE, MAX_POSTS_PER_RUN

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
GEMINI_IMAGE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_IMAGE_MODEL}:generateContent"

TELEGRAM_SEND_MESSAGE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
TELEGRAM_SEND_PHOTO = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

POSTED_IDS_FILE = os.path.join(os.path.dirname(__file__), "posted_ids.json")

MIN_WORDS_IN_POST = 35
MAX_REWRITE_ATTEMPTS = 2  # birinchi urinish + 1 marta qayta yozish


def check_env():
    missing = [
        name for name, val in [
            ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
            ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
            ("GROQ_API_KEY", GROQ_API_KEY),
            ("GEMINI_API_KEY", GEMINI_API_KEY),
        ] if not val
    ]
    if missing:
        print(f"XATOLIK: quyidagi environment variable'lar berilmagan: {missing}")
        sys.exit(1)


def load_posted_ids():
    if os.path.exists(POSTED_IDS_FILE):
        try:
            with open(POSTED_IDS_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_posted_ids(ids):
    ids_list = list(ids)[-2000:]
    with open(POSTED_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(ids_list, f, ensure_ascii=False, indent=2)


def make_id(entry):
    key = entry.get("link") or entry.get("id") or entry.get("title", "")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def call_groq(prompt, max_tokens=400, temperature=0.4):
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq xatoligi: {e}")
        return None


def rewrite_with_ai(title, summary, source_name, extra_feedback=None):
    feedback_line = f"\n\nEslatma: {extra_feedback}" if extra_feedback else ""
    prompt = (
        "Sen professional o'zbek jurnalistisan. Quyidagi yangilikni o'zbek tilida, "
        "aniq, neytral va qisqa (80-120 so'z) uslubda qayta yoz. Faktlarni o'zgartirma, "
        "shaxsiy fikr qo'shma, sarlavha va matnni tabiiy jurnalistik uslubda ber. "
        "Javobda faqat tayyor post matnini ber (izohsiz), oxirida mos bitta emoji qo'y."
        f"{feedback_line}\n\n"
        f"Manba: {source_name}\n"
        f"Sarlavha: {title}\n"
        f"Qisqacha mazmuni: {summary}"
    )
    return call_groq(prompt)


def quality_check(post_text, source_title):
    if not post_text or len(post_text.split()) < MIN_WORDS_IN_POST:
        return False, "Post juda qisqa yoki bo'sh"

    prompt = (
        "Sen sifat nazorati bo'yicha tahririyat muharririsan. Quyidagi tayyor post "
        "matnini tekshir. Post quyidagi shartlarga javob bersa, faqat bitta so'z bilan "
        "'HA' deb javob ber:\n"
        "1) Mazmunan tushunarli, to'liq gaplardan iborat va grammatik jihatdan to'g'ri\n"
        "2) Asl sarlavhaga mos va faktlarga zid emas\n"
        "3) Spam, reklama yoki mazmunsiz matn emas\n"
        "4) Haqoratli, nafrat uyg'otuvchi yoki nomaqbul kontent yo'q\n"
        "5) Faqat o'zbek tilida yozilgan\n\n"
        "Agar shartlardan birortasi buzilgan bo'lsa, 'YOQ: ' deb boshlab qisqa sababini yoz.\n\n"
        f"Asl sarlavha: {source_title}\n\n"
        f"Tekshiriladigan post:\n{post_text}"
    )
    result = call_groq(prompt, max_tokens=100, temperature=0.1)
    if result and result.strip().upper().startswith("HA"):
        return True, None
    return False, (result or "AI javob bermadi")


def get_quality_checked_post(title, summary, source_name):
    feedback = None
    for attempt in range(1, MAX_REWRITE_ATTEMPTS + 1):
        text = rewrite_with_ai(title, summary, source_name, extra_feedback=feedback)
        if not text:
            continue
        ok, reason = quality_check(text, title)
        if ok:
            return text
        print(f"  Sifat nazoratidan o'tmadi (urinish {attempt}): {reason}")
        feedback = f"Oldingi urinish rad etildi: {reason}. Buni tuzatib qayta yoz."
