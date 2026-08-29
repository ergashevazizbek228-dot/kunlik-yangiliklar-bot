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

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
GEMINI_IMAGE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_IMAGE_MODEL}:generateContent"

TELEGRAM_SEND_MESSAGE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
TELEGRAM_SEND_PHOTO = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

POSTED_IDS_FILE = os.path.join(os.path.dirname(__file__), "posted_ids.json")

MIN_WORDS_IN_POST = 35
MAX_REWRITE_ATTEMPTS = 2  # birinchi urinish + 1 marta qayta yozish


# ------------------------------------------------------------------
# Yordamchi: fayl bilan ishlash
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Groq (matnni qayta yozish + sifat nazorati)
# ------------------------------------------------------------------

def call_groq(prompt, max_tokens=400, temperature=0.4):
    last_error = None
    for model in GROQ_MODELS:
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                last_error = f"model={model} status={resp.status_code} body={resp.text[:300]}"
                print(f"Groq xatoligi ({model}): {resp.status_code} - {resp.text[:300]}")
                continue
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_error = f"model={model} exception={e}"
            print(f"Groq istisnosi ({model}): {e}")
            continue
    print(f"Barcha Groq modellari ishlamadi. Oxirgi xatolik: {last_error}")
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
    """Postni AI orqali sifat nazoratidan o'tkazadi."""
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
    """Post yozadi va sifat nazoratidan o'tkazadi, kerak bo'lsa qayta yozadi."""
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
    return None


# ------------------------------------------------------------------
# Rasm topish / generatsiya qilish
# ------------------------------------------------------------------

def extract_image_from_entry(entry):
    """RSS xabarining o'zidan rasm havolasini topishga harakat qiladi."""
    try:
        if getattr(entry, "media_content", None):
            url = entry.media_content[0].get("url")
            if url:
                return url
        if getattr(entry, "media_thumbnail", None):
            url = entry.media_thumbnail[0].get("url")
            if url:
                return url
        for link in entry.get("links", []):
            if str(link.get("type", "")).startswith("image"):
                return link.get("href")
    except Exception:
        pass
    return None


def extract_og_image(article_url):
    """Maqola sahifasidan og:image meta-tegini o'qiydi."""
    if not article_url:
        return None
    try:
        resp = requests.get(
            article_url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"},
            allow_redirects=True,
        )
        match = re.search(
            r'<meta[^>]+(?:property|name)=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            resp.text, re.IGNORECASE,
        )
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def generate_image_ai(title, post_text):
    """Gemini 2.5 Flash Image (Nano Banana) orqali post uchun rasm yaratadi."""
    image_prompt = (
        "Create a realistic, editorial-style photograph illustrating the concept of the "
        "following news topic. No text, letters, or watermarks in the image. "
        "Do not depict any real, identifiable named person — use symbolic, conceptual, "
        "or generic imagery instead (objects, locations, abstract representations). "
        "Professional, neutral news-photography style.\n\n"
        f"Topic: {title}\n{post_text[:300]}"
    )
    try:
        resp = requests.post(
            f"{GEMINI_IMAGE_URL}?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": image_prompt}]}]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    except Exception as e:
        print(f"  Rasm generatsiya xatoligi: {e}")
    return None


# ------------------------------------------------------------------
# Telegram'ga joylash
# ------------------------------------------------------------------

def build_caption(text, link):
    caption = f"{text}\n\n🔗 Manba: {link}"
    if len(caption) > 1024:
        cut = 1024 - len(f"...\n\n🔗 Manba: {link}") - 3
        caption = f"{text[:max(cut, 0)]}...\n\n🔗 Manba: {link}"
        caption = caption[:1024]
    return caption


def post_photo_url(image_url, caption):
    resp = requests.post(
        TELEGRAM_SEND_PHOTO,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "photo": image_url,
            "caption": caption,
            "parse_mode": "HTML",
        },
        timeout=30,
    )
    return resp.status_code == 200, resp.text


def post_photo_bytes(image_bytes, caption):
    resp = requests.post(
        TELEGRAM_SEND_PHOTO,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML",
        },
        files={"photo": ("news.png", image_bytes, "image/png")},
        timeout=60,
    )
    return resp.status_code == 200, resp.text


def post_text_only(caption):
    resp = requests.post(
        TELEGRAM_SEND_MESSAGE,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": caption,
            "parse_mode": "HTML",
        },
        timeout=20,
    )
    return resp.status_code == 200, resp.text


def publish_post(title, text, link, entry):
    """Rasm bilan (yoki bo'lmasa AI generatsiya qilingan rasm bilan) postni joylaydi."""
    caption = build_caption(text, link)

    # 1) Manbadagi haqiqiy rasmni sinab ko'ramiz
    image_url = extract_image_from_entry(entry) or extract_og_image(link)
    if image_url:
        ok, info = post_photo_url(image_url, caption)
        if ok:
            return True
        print(f"  Manba rasmi ishlamadi ({info[:120]}), AI bilan rasm yaratamiz...")

    # 2) AI orqali rasm generatsiya qilamiz (Nano Banana / Gemini)
    image_bytes = generate_image_ai(title, text)
    if image_bytes:
        ok, info = post_photo_bytes(image_bytes, caption)
        if ok:
            return True
        print(f"  Generatsiya qilingan rasmni yuborib bo'lmadi: {info[:120]}")

    # 3) Oxirgi chora — faqat matn bilan yuboramiz
    print("  Rasmsiz, faqat matn bilan yuborilmoqda.")
    ok, info = post_text_only(caption)
    if not ok:
        print(f"  Matnli post ham yuborilmadi: {info[:120]}")
    return ok


# ------------------------------------------------------------------
# Asosiy jarayon
# ------------------------------------------------------------------

def main():
    check_env()
    posted_ids = load_posted_ids()
    new_posts_count = 0

    for source_name, feed_url in SOURCES:
        if new_posts_count >= MAX_POSTS_PER_RUN:
            break

        print(f"Tekshirilmoqda: {source_name}")
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"  Feed o'qishda xatolik: {e}")
            continue

        entries = feed.entries[:ENTRIES_PER_SOURCE]

        for entry in entries:
            if new_posts_count >= MAX_POSTS_PER_RUN:
                break

            entry_id = make_id(entry)
            if entry_id in posted_ids:
                continue

            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            link = entry.get("link", "")

            post_text = get_quality_checked_post(title, summary, source_name)
            if not post_text:
                print(f"  O'tkazib yuborildi (sifat nazoratidan o'tmadi): {title[:60]}")
                posted_ids.add(entry_id)  # qayta-qayta urinib o'tirmaslik uchun
                continue

            success = publish_post(title, post_text, link, entry)
            if success:
                print(f"  Joylandi: {title[:60]}")
                posted_ids.add(entry_id)
                new_posts_count += 1
                time.sleep(3)  # Telegram flood-limitdan saqlanish

    save_posted_ids(posted_ids)
    print(f"Tugadi. Jami yangi post: {new_posts_count}")


if __name__ == "__main__":
    main()
