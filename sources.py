# -*- coding: utf-8 -*-
"""
Yangiliklar manbalari.
Google News RSS'dan foydalanamiz — bu eng barqaror bepul usul,
chunki har bir saytning o'z RSS manzilini qidirib yurishga hojat yo'q,
va u mavzu bo'yicha qidiruv qilib beradi.

Har bir manba: (nomi, RSS url)
Xohlasangiz bu ro'yxatga o'zingiz istagan mavzu yoki saytni qo'shishingiz mumkin.
"""

def google_news_rss(query, lang="uz", country="UZ"):
    """Google News RSS qidiruv havolasini quradi."""
    query = query.replace(" ", "%20")
    return f"https://news.google.com/rss/search?q={query}&hl={lang}&gl={country}&ceid={country}:{lang}"


SOURCES = [
    # --- O'zbekiston yangiliklari ---
    ("O'zbekiston umumiy", google_news_rss("O'zbekiston yangiliklari", "uz", "UZ")),
    ("O'zbekiston iqtisodiyot", google_news_rss("O'zbekiston iqtisodiyot", "uz", "UZ")),

    # --- Dunyo yangiliklari ---
    ("Dunyo yangiliklari", google_news_rss("world news", "en", "US")),

    # --- Texnologiya va AI ---
    ("Texnologiya", google_news_rss("technology", "en", "US")),
    ("Sun'iy intellekt (AI)", google_news_rss("artificial intelligence", "en", "US")),

    # --- Sport ---
    ("Sport", google_news_rss("sport yangiliklari", "uz", "UZ")),
]

# Har bir manbadan bir marta ishga tushganda nechta eng yangi xabar tekshirilsin
ENTRIES_PER_SOURCE = 5

# Bir marta ishga tushganda kanalga eng ko'pi bilan nechta post joylansin
MAX_POSTS_PER_RUN = 5
