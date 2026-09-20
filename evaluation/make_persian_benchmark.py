#!/usr/bin/env python3
"""ARION ALPHA 1 — Persian fluency benchmark generator (deterministic).

Produces evaluation/persian_fluency_benchmark.jsonl with 200+ prompts across
12 categories (Persian-phase spec). Hand-authored core prompts for the
precision-critical cases (language switching, Arabic-vs-Persian) plus
systematic domain variants for breadth.

Entry schema:
  {"id", "category", "expect_lang", "prompt", "history": [...optional...], "note"}
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "evaluation", "persian_fluency_benchmark.jsonl")

P = []  # (category, expect_lang, prompt, history|None, note)

# ---------------- language switching (precision-critical) ----------------
P += [
    ("language_switching", "fa", "به فارسی جواب بده: توضیح بده DNS چیست.", None, "explicit fa request"),
    ("language_switching", "fa", "من فارسی صحبت می‌کنم. فقط فارسی جواب بده. فرق GET و POST چیه؟", None, "explicit fa-only"),
    ("language_switching", "ar", "به عربی جواب بده.", None, "explicit ar request"),
    ("language_switching", "ar", "فقط به عربی جواب بده: ما هي الحاسوب؟", None, "ar question + ar instruction"),
    ("language_switching", "en", "Answer in English from now on. What is normalization in databases?", None, "explicit en request"),
    ("language_switching", "fa", "فارسی جواب بده. هوش مصنوعی چیه؟", None, "short fa instruction"),
    ("language_switching", "fa", "از این به بعد فقط فارسی بنویس. مزایای لینوکس چیه؟", None, "fa-only constraint"),
    ("language_switching", "en", "اگه اجازه بدی این سوال رو انگلیسی بپرسم: why is the sky blue?", None, "user switches to en"),
    ("language_switching", "fa", "برگرد به فارسی. داشبورد ادمین چی؟", None, "switch back to fa"),
    ("language_switching", "fa", "جواب‌هات رو از این لحظه به زبان فارسی ارائه بده.", None, "formal fa instruction"),
    ("language_switching", "ar", "Please answer in Arabic: what is a variable?", None, "en prompt asking for ar"),
    ("language_switching", "fa", "Answer in Persian/Farsi: what is machine learning?", None, "en prompt asking for fa"),
]

# ---------------- arabic vs persian distinction ----------------
P += [
    ("arabic_vs_persian", "fa", "شعر از حافظ بخوان.", None, "fa classic"),
    ("arabic_vs_persian", "fa", "من فارسی صحبت می‌کنم نه عربی. لطفاً فارسی جواب بده: وردپرس چیه؟", None, "anti-arabic confusion"),
    ("arabic_vs_persian", "fa", "این جمله به چه زبانی است؟ «کتاب» و «قلم» چه معنایی دارند؟", None, "shared words — answer fa"),
    ("arabic_vs_persian", "ar", "ما هو الفرق بين القهوة والشاي؟ أجب بالعربية.", None, "real ar request"),
    ("arabic_vs_persian", "fa", "زبان من فارسی است. دربارهٔ تاریخ ایران توضیح بده.", None, "fa identity"),
    ("arabic_vs_persian", "fa", "کلمهٔ «عشق» در جملهٔ فارسی استفاده کن.", None, "fa output"),
    ("arabic_vs_persian", "ar", "اشرح لي ما هو الإنترنت بالعربية.", None, "ar output"),
    ("arabic_vs_persian", "fa", "دو خط پشت سر هم بنویس: اولی فارسی، دومی عربی — موضوع: هوا.", None, "both languages, fa first"),
    ("arabic_vs_persian", "fa", "فارسی و عربی چه فرق‌هایی از نظر حروف دارند؟ جواب فارسی بده.", None, "meta question, fa answer"),
]

# ---------------- casual conversation ----------------
P += [
    ("casual_conversation", "fa", "سلام", None, "greeting"),
    ("casual_conversation", "fa", "چطوری؟", None, "how are you"),
    ("casual_conversation", "fa", "حالت چطوره امروز؟", None, "smalltalk"),
    ("casual_conversation", "fa", "یه جوک بگو.", None, "joke"),
    ("casual_conversation", "fa", "خسته‌ام از کار، چی کار کنم؟", None, "venting"),
    ("casual_conversation", "fa", "اسم من علیه، دوست داری باهم آشنا بشیم؟", None, "name intro"),
    ("casual_conversation", "fa", "روزت چطور گذشت؟", None, "day talk"),
    ("casual_conversation", "fa", "من امروز قبول شدم! 🎉", None, "celebration"),
    ("casual_conversation", "fa", "چی کارار دریاری؟", None, "colloquial filler"),
    ("casual_conversation", "fa", "برای آخر هفته چه پیشنهادی داری؟", None, "weekend plans"),
    ("casual_conversation", "fa", "دلم گرفته، یه چیزی بگو حال خوب بشه.", None, "mood"),
    ("casual_conversation", "fa", "با چند تا دوست قراره بریم شمال، لیست وسایل لازم رو بگو.", None, "trip planning"),
]

# ---------------- colloquial persian ----------------
P += [
    ("colloquial_persian", "fa", "خودمونی توضیح بده داکر چیه.", None, "docker colloquial"),
    ("colloquial_persian", "fa", "ساده بگو، بیت‌کوین با بانک چه فرقی داره؟", None, "bitcoin colloquial"),
    ("colloquial_persian", "fa", "به زبون خودمونی بگو API یعنی چی.", None, "api colloquial"),
    ("colloquial_persian", "fa", "مثل یه بچه ۵ ساله توضیح بده اینترنت چیه.", None, "ELI5"),
    ("colloquial_persian", "fa", "راستش رو بگو، یادگیری برنامه‌نویسی سخت‌ه؟", None, "honest colloquial"),
    ("colloquial_persian", "fa", "بهم بگو اینترنت اکسپلورر چرا مرد؟", None, "casual tech history"),
    ("colloquial_persian", "fa", "خودمونی بگو وای‌فای چطور کار می‌کنه.", None, "wifi colloquial"),
    ("colloquial_persian", "fa", "یه مثال بزن برای چی هستن کوکی‌های مرورگر.", None, "cookies colloquial"),
    ("colloquial_persian", "fa", "عربی حرف نزن، خودمونی فارسی بگو وایرشارک چیه.", None, "colloquial + anti-ar"),
    ("colloquial_persian", "fa", "خلاصه و مفید بگو، ضد ویروس لازمه برای لینوکس؟", None, "short colloquial"),
]

# ---------------- formal persian ----------------
P += [
    ("formal_persian", "fa", "رسمی توضیح بده مفهوم امنیت اطلاعات را.", None, "formal register"),
    ("formal_persian", "fa", "یک نامهٔ اداری رسمی برای مرخصی استحقاقی بنویسید.", None, "formal letter"),
    ("formal_persian", "fa", "خواهشمند است معایب و مزایای کار از راه دور را به صورت رسمی بیان کنید.", None, "formal analysis"),
    ("formal_persian", "fa", "لطفاً به زبان اداری، یک اطلاعیهٔ تغییر ساعت کاری شرکت بنویسید.", None, "formal notice"),
    ("formal_persian", "fa", "با ادبیات رسمی دربارهٔ اهمیت آموزش مداوم بنویسید.", None, "formal essay"),
    ("formal_persian", "fa", "رسمی و مودبانه از مشتری بخواهید فاکتور پرداخت‌نشده را تسویه کند.", None, "polite request"),
    ("formal_persian", "fa", "به رسمی‌ترین شکل ممکن، علت تأخیر پروژه را به مدیرعامل گزارش کنید.", None, "formal report"),
]

# ---------------- technical persian ----------------
P += [
    ("technical_persian", "fa", "REST API چیست و چه تفاوتی با GraphQL دارد؟", None, "api concepts"),
    ("technical_persian", "fa", "دیتابیس رابطه‌ای و غیررابطه‌ای را مقایسه کنید.", None, "db compare"),
    ("technical_persian", "fa", "برای وب‌سایت فروشگاهی، MongoDB مناسب‌تره یا PostgreSQL؟ چرا؟", None, "db choice"),
    ("technical_persian", "fa", "مفهوم کش (cache) را در وب توضیح دهید.", None, "caching"),
    ("technical_persian", "fa", "CDN چه کمکی به سرعت سایت ایرانی می‌کند؟", None, "cdn"),
    ("technical_persian", "fa", "مراحل deploy یک سایت React روی سرور را بگو.", None, "deployment"),
    ("technical_persian", "fa", "تفاوت فریم‌ورک و کتابخانه در برنامه‌نویسی چیست؟", None, "framework vs lib"),
    ("technical_persian", "fa", "مجازی‌سازی با داکر چه مزیت‌هایی نسبت به VM سنتی دارد؟", None, "docker vs vm"),
    ("technical_persian", "fa", "پروتکل HTTPS چطور امنیت را تأمین می‌کند؟", None, "https"),
    ("technical_persian", "fa", "Git Flow و Trunk-based development را مقایسه کنید.", None, "git workflows"),
]

# ---------------- persian coding ----------------
P += [
    ("persian_coding", "fa", "این خط JavaScript چرا error میده؟\n```js\nconst res = await fetch('/api/users');\nconst data = res.json();\n```", None, "debug: missing await"),
    ("persian_coding", "fa", "چرا این کد پایتون کار نمی‌کنه؟\n```python\nfor i in range(len(items)):\n    if items[i] > 5: del items[i]\n```", None, "debug: list mutation"),
    ("persian_coding", "fa", "یه تابع پایتون بنویس که معکوس یک رشته رو برگردونه.", None, "code gen py"),
    ("persian_coding", "fa", "چطور با CSS یه دیو رو هم عمودی هم افقی وسط بچینم؟", None, "css centering"),
    ("persian_coding", "fa", "این کوئری SQL چه مشکلی داره؟\n```sql\nSELECT * FROM users WHERE name = \" + userInput + \"\n```", None, "sql injection debug"),
    ("persian_coding", "fa", "یه فرم HTML با اعتبارسنجی ساده جاوااسکریپت بنویس.", None, "form gen"),
    ("persian_coding", "fa", "چرا این دستور git کار نمی‌کنه: `git push origin master` خطای rejected می‌ده؟", None, "git help"),
    ("persian_coding", "fa", "در React فرق useState و useEffect رو با مثال توضیح بده.", None, "react hooks"),
    ("persian_coding", "fa", "چطور در ترمینال لینوکس فایل‌های بزرگ‌تر از ۱۰۰ مگابایت رو پیدا کنم؟", None, "linux find"),
    ("persian_coding", "fa", "این ارور Node.js یعنی چی: «Cannot read properties of undefined»؟", None, "node error explain"),
]

# ---------------- explanations ----------------
P += [
    ("explanations", "fa", "ساده توضیح بده: یادگیری ماشین چیه؟", None, "ML simple"),
    ("explanations", "fa", "فرق حافظهٔ RAM و حافظهٔ ذخیره‌سازی چیه؟", None, "ram vs disk"),
    ("explanations", "fa", "دنیای دیجیتال، رمزنگاری متقارن و نامتقارن رو توضیح بده.", None, "crypto explain"),
    ("explanations", "fa", "چرا آسمون آبی دیده می‌شه؟", None, "science"),
    ("explanations", "fa", "تورم دقیقاً یعنی چی و چرا پول کم‌ارزش می‌شه؟", None, "economics"),
    ("explanations", "fa", "کوئری SQL با JOIN چطور کار می‌کنه؟ با مثال ساده بگو.", None, "sql join"),
    ("explanations", "fa", "این یعنی چی؟ «الگوریتم زمان چندضلعی دارد».", None, "big-o explain"),
    ("explanations", "fa", "برق ساختمان چطور از پست به خونه ما میاد؟", None, "power grid"),
]

# ---------------- summarization ----------------
P += [
    ("summarization", "fa", "این متن رو در دو خط خلاصه کن: «هوش مصنوعی در سال‌های اخیر پیشرفت چشمگیری داشته است. مدل‌های زبانی بزرگ می‌توانند متن تولید کنند و به سؤالات پاسخ دهند. با این حال گاهی اطلاعات نادرست تولید می‌کنند که به آن توهم می‌گویند. پس خروجی آن‌ها باید بررسی شود.»", None, "summarize tech"),
    ("summarization", "fa", "خلاصهٔ خبر: «بانک مرکزی اعلام کرد نرخ سود سپرده‌های یک‌ساله unchanged می‌ماند و تورم سالانه در فصل تابستان نسبت به بهار کمی کاهش یافته است. کارشناسان معتقدند ادامهٔ این روند به ثبات بازار کمک می‌کند.»", None, "summarize news"),
    ("summarization", "fa", "این پاراگراف رو تو یه جمله بگو: «برای داشتن خواب خوب بهتر است قبل از خواب از گوشی استفاده نکنید، اتاق را تاریک و خنک نگه دارید و هر روز در ساعت مشخصی بخوابید.»", None, "one-line summary"),
]

# ---------------- rewriting ----------------
P += [
    ("rewriting", "fa", "خودمونی بازنویسی کن: «توصیه می‌گردد پیش از به‌روزرسانی، از اطلاعات خود نسخهٔ پشتیبان تهیه فرمایید.»", None, "formal→casual"),
    ("rewriting", "fa", "رسمی بازنویسی کن: «سلام، پولت رو که ندادی بفرست دیگه!»", None, "casual→formal"),
    ("rewriting", "fa", "این متن انگلیسی رو به فارسی روان ترجمه کن: \"Clean code is code that is easy to understand and change.\"", None, "translate en→fa"),
    ("rewriting", "fa", "این جملهٔ فارسی رو به انگلیسی ترجمه کن: «تجربهٔ کاربری خوب، فروش را افزایش می‌دهد.»", None, "translate fa→en"),
    ("rewriting", "fa", "طوری بازنویسی کن که برای بچه‌ها قابل فهم باشه: «فوتوسنتز فرایند تولید انرژی در گیاهان با استفاده از نور خورشید است.»", None, "simplify for kids"),
]

# ---------------- multi-turn ----------------
P += [
    ("multi_turn", "fa", "حالا کاملش کن — ساعت کاری و شماره تماس هم بذار.",
     [{"role": "user", "content": "یه سایت برای نانوایی می‌خوام."},
      {"role": "assistant", "content": "حتماً! چه سبکی مدنظرت هست؟ مدرن یا کلاسیک؟ منو محصول هم می‌خوای؟"},
      {"role": "user", "content": "مدرن باشه و فارسی و راست‌چین."}], "requirement evolution"),
    ("multi_turn", "fa", "نه، منظورم این نبود — رنگ پس‌زمینه رو نمی‌خوام عوض کنم، رنگ متن دکمه رو می‌گم.",
     [{"role": "user", "content": "چطور رنگ دکمه رو عوض کنم؟"},
      {"role": "assistant", "content": "با CSS: `background-color` رو در استایل دکمه تغییر بده. مثلاً `background-color: #22c55e;`"}], "correction handling"),
    ("multi_turn", "fa", "خب حالا همونو با دیتابیس وصل کن.",
     [{"role": "user", "content": "یه فرم تماس ساده HTML بده."},
      {"role": "assistant", "content": "بفرما: `<form><input name=\"name\"><textarea name=\"msg\"></textarea><button>ارسال</button></form>`"}], "iterative build"),
    ("multi_turn", "fa", "ساده‌ترش کن، زیاد پیچیده شد.",
     [{"role": "user", "content": "یه داشبورد برای فروشگاه با نمودار فروش بساز."},
      {"role": "assistant", "content": "این داشبورد با Chart.js شامل نمودار خطی فروش ماهانه، کارت‌های KPI و جدول پرفروش‌ترین محصولات است: [کد کامل ۸۰ خطی]"}], "simplification request"),
    ("multi_turn", "fa", "راستی کد قبلی رو می‌خوام با TypeScript باشه.",
     [{"role": "user", "content": "یه تابع جاوااسکریپت برای اعتبارسنجی ایمیل بده."},
      {"role": "assistant", "content": "بفرما:\n```js\nfunction isEmail(s) {\n  return /^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$/.test(s);\n}\n```"}], "retroactive change"),
]

# ---------------- persian + english mixed ----------------
P += [
    ("persian_english", "fa", "برای deploy پروژهٔ React‌مون Caddy بهتره یا nginx؟", None, "fa + tech terms"),
    ("persian_english", "fa", "فرق بین commit و push در git چیه؟", None, "fa + git terms"),
    ("persian_english", "fa", "وقتی می‌گن لایبرری deprecated شده یعنی چی؟", None, "fa + english word"),
    ("persian_english", "fa", "این جملهٔ فنی رو طبیعی فارسی کن: \"We need to scale the backend before the launch.\"", None, "translate to natural fa"),
    ("persian_english", "fa", "چرا وقتی bundle size زیاده لود سایت کند می‌شه؟", None, "fa + bundler terms"),
    ("persian_english", "fa", "با Express یه API ساده برای لیست کتاب‌ها بنویس، توضیحش فارسی.", None, "code + fa explanation"),
    ("persian_english", "fa", "احتمالاً مشکل از CORS ـه. دقیقاً چیه این؟", None, "fa + CORS"),
    ("persian_english", "fa", "داشبورد رو با charts مخلوط کن ولی performance رو حفظ کن — چیکار کنم؟", None, "fa + tech verbs"),
]

# ---------------- systematic domain variants (breadth) ----------------
DOMAINS_FA = [
    ("آشپزی", "دستور پخت قرمه‌سبزی رو بگو", "طرز تهیهٔ کیک اسفنجی رو قدم‌به‌قدم بگو", "بهترین ادویه برای خورش قیمه چیه؟"),
    ("ورزش", "برای مبتدی‌ها برنامهٔ تمرین باشگاه بده", "چطور دونده بهتری بشم؟", "قبل از ورزش باید چی بخورم؟"),
    ("سفر", "برای سفر به مشهد چه نکاتی مهمه؟", "ارزون‌ترین وقت سفر به کیش کیه؟", "چمدون سفر زمستونی چی ببرم؟"),
    ("تحصیل", "برای کنکور تجربی برنامهٔ مطالعه بچین", "چطور تمرکزم رو موقع درس خوندن بالا ببرم؟", "بهترین روش یادگیری زبان انگلیسی چیه؟"),
    ("مالی", "پس‌انداز رو کجا بذارم که ارزشش کم نشه؟", "فرق صندوق سرمایه‌گذاری با سپردهٔ بانکی چیه؟", "چطور بودجهٔ ماهانه تنظیم کنم؟"),
    ("سلامت", "برای کم‌خوابی چه راه‌حل طبیعی هست؟", "ویتامین D چه فایده‌ای داره؟", "کمردرد نشستن پشت میز رو چطور درمان کنم؟"),
    ("فناوری", "بهترین گوشی زیر ۲۰ میلیون تومن چیه؟", "لپ‌تاپ برای برنامه‌نویسی چه مشخصاتی لازم داره؟", "تفاوت آیپد و تبلت معمولی چیه؟"),
    ("خانه", "برای دکوراسیون آشپزخانه کوچیک چه ایده‌ای داری؟", "چطور از گیاهان آپارتمانی مراقبت کنم؟", "رنگ مناسب اتاق خواب چی باشه؟"),
    ("فرهنگ", "چند کتاب خوب ایرانی برای شروع کتاب‌خوندن معرفی کن", "سینمای ایران چه filmmaker های معروفی داشته؟", "تقویم نوروزی چه آیین‌هایی داره؟"),
    ("کار", "برای مصاحبهٔ شغلی چطور آماده بشم؟", "رزومهٔ خوب چه ویژگی‌هایی داره؟", "چطور با همکار سخت کنار بیام؟"),
]
STYLE_TAGS = ["", " (ساده توضیح بده)", " (خودمونی)", " (رسمی توضیح بده)", " — یه مثال هم بزن"]

i = 0
for dom, q1, q2, q3 in DOMAINS_FA:
    for q in (q1, q2, q3):
        for style in (STYLE_TAGS[0], STYLE_TAGS[1 + i % 4]):
            cat = "formal_persian" if "رسمی" in style else (
                "colloquial_persian" if "خودمونی" in style else (
                    "explanations" if "مثال" in style or "ساده" in style else "casual_conversation"))
            P.append((cat if cat != "casual_conversation" else "explanations",
                      "fa", q + style, None, f"domain variant: {dom}"))

# ---------------- extra systematic variants (breadth, second wave) ----------------
DOMAINS_FA2 = [
    ("خرید آنلاین", "چطور از جنس تقلبی توی فروشگاه‌های اینترنتی جلوگیری کنم؟", "مقایسه قیمت بهترین روشش چیه؟", "شرایط مرجوع کردن کالا در دیجی‌کالا چیه؟"),
    ("آموزش کودک", "برای کودک ۵ ساله چه بازی‌های آموزشی خوبه؟", "چطور بچه رو به کتاب‌خوندن علاقه‌مند کنم؟", "زمان مناسب شروع یادگیری زبان دوم چیه؟"),
    ("عکاسی", "برای عکاسی با موبایل چه ترفندهایی هست؟", "دیافراگم دقیقاً چه کاری انجام می‌ده؟", "برای شروع عکاسی چه دوربینی بخرم؟"),
    ("موسیقی", "برای شروع گیتار از کجا شروع کنم؟", "نت‌خوندی چقدر برای نوازندگی ضروریه؟", "بهترین ساز برای بچه ۸ ساله چیه؟"),
    ("امنیت", "چطور پسورد قوی بسازم و یادم بمونه؟", "دو مرحله‌ای کردن اکانت چطوریه؟", "فیشینگ چیه و چطور بشناسمش؟"),
    ("کارآفرینی", "برای شروع استارتاپ اولین قدم چیه؟", "چطور ایده‌م رو به سرمایه‌دار معرفی کنم؟", "بیزنس‌پلن خوب چه بخش‌هایی داره؟"),
    ("درمانگاه", "فرق دکتر داخلی و عمومی چیه؟", "چکاپ سالانه شامل چه آزمایش‌هایی می‌شه؟", "برای ویزیت آنلاین چه اپی‌هایی خوبه؟"),
    ("برنامه‌نویسی پیشرفته", "معماری مونولیت و میکروسرویس رو مقایسه کن", "الگوی طراحی Singleton کی استفاده می‌شه؟", "TDD دقیقاً چه فرایندی داره؟"),
    ("سئو", "برای بالا اومدن تو گوگل چیکار کنم؟", "بک‌لینک چیه و چقدر مهمه؟", "سرعت سایت چقدر روی سئو اثر داره؟"),
    ("هوش مصنوعی", "ChatGPT چطور کار می‌کنه به زبان ساده؟", "فرق machine learning و deep learning چیه؟", "مدل زبانی چیست و چه کاربردهایی دارد؟"),
]
for dom, q1, q2, q3 in DOMAINS_FA2:
    for q in (q1, q2, q3):
        for style in (STYLE_TAGS[0], STYLE_TAGS[2 if q1 is q else 3]):
            cat = "formal_persian" if "رسمی" in style else (
                "colloquial_persian" if "خودمونی" in style else "explanations")
            P.append((cat, "fa", q + style, None, f"domain variant: {dom}"))

# extra multi-turn and language-switch breadth
P += [
    ("multi_turn", "fa", "مرسی! حالا سومین بخش رو هم اضافه کن.",
     [{"role": "user", "content": "یه landing page برای استارتاپ می‌خوام."},
      {"role": "assistant", "content": "بفرما — هدر، بخش ویژگی‌ها و فرم تماس رو گذاشتم. می‌خوای بخش قیمت‌گذاری هم اضافه کنم؟"},
      {"role": "user", "content": "آره قیمت‌گذاری هم بذار."}], "multi-step build"),
    ("multi_turn", "fa", "ایول! الان موبایل رو چک کردم، منو باز نمی‌شه.",
     [{"role": "user", "content": "یه منوی ناوبری بساز."},
      {"role": "assistant", "content": "این منوی افقی با HTML و CSS: `<nav><a>خانه</a><a>محصولات</a><a>تماس</a></nav>`"}], "bug report flow"),
    ("multi_turn", "fa", "باشه ولی حالا رنگ‌بندیش رو آبی کن.",
     [{"role": "user", "content": "یه کارت محصول بساز."},
      {"role": "assistant", "content": "این کارت محصول با پس‌زمینهٔ سبز و قیمت برجسته: [کد]"}], "style change"),
    ("multi_turn", "fa", "چرا هنوز کار نمی‌کنه؟ همون خطا رو می‌ده.",
     [{"role": "user", "content": "کدم خطای 500 می‌ده."},
      {"role": "assistant", "content": "خطای 500 یعنی مشکل سمت سرور. لاگ سرور رو چک کن و محتوای خطا رو برام بفرست."}], "follow-up debugging"),
    ("language_switching", "fa", "دو زبونه جواب بده: اول فارسی بعد انگلیسی — سؤال: بازخورد (feedback) چیه؟", None, "bilingual answer"),
    ("language_switching", "fa", "از اینجا به بعد مکالمه‌مون فارسی باشه.", None, "fa continuation"),
    ("language_switching", "en", "از حالا انگلیسی جواب بده لطفاً. What is docker compose?", None, "fa request → en answer"),
    ("language_switching", "ar", "لا أتحدث الفارسية، أجب بالعربية: ما هو الهاتف الذكي؟", None, "ar-only user"),
    ("language_switching", "fa", "صبر کن، از اول فارسی صحبت کردیم نه؟ خب ادامه بده فارسی.", None, "meta fa check"),
    ("casual_conversation", "fa", "چه خبر؟", None, "whats up"),
    ("casual_conversation", "fa", "فردا تعطیلی‌م، برنامه چیکار کنم؟", None, "day off"),
    ("casual_conversation", "fa", "یه فیلم ایرانی پیشنهاد بده.", None, "movie rec"),
    ("casual_conversation", "fa", "قسمت این داستان اینه که دوستم رفته مهاجرت، حس خوبی ندارم.", None, "emotional support"),
    ("colloquial_persian", "fa", "بگو ببینم، قرعه‌کشی بانکی واقعیه یا کلاهبرداری؟", None, "colloquial skepticism"),
    ("colloquial_persian", "fa", "موبایل قدیمی‌م بریزه به آب! تعویض کنم یا تعمیر؟", None, "colloquial advice"),
    ("formal_persian", "fa", "متنی رسمی دربارهٔ اهمیت محیط‌زیست بنویسید.", None, "formal env"),
    ("technical_persian", "fa", "برای بکاپ‌گیری خودکار از دیتابیس چه راه‌حلی پیشنهاد می‌کنید؟", None, "backup strategy"),
    ("technical_persian", "fa", "تفاوت HTTP/1.1 و HTTP/2 در چیست؟", None, "protocol compare"),
    ("persian_coding", "fa", "با پایتون یه اسکریپت بنویس که فایل‌های تکراری رو تو یه پوشه پیدا کنه.", None, "py script gen"),
    ("persian_coding", "fa", "چرا این کد CSS اعمال نمی‌شه؟\n```css\np {\n  color: red\n  font-size: 14px;\n}\n```", None, "css syntax bug"),
]
ids = []
out_n = 0
for idx, (cat, lang, prompt, hist, note) in enumerate(P):
    rec = {
        "id": f"fa-bench-{idx:04d}",
        "category": cat,
        "expect_lang": lang,
        "prompt": prompt,
        "note": note,
    }
    if hist:
        rec["history"] = hist
    ids.append(rec)
    out_n += 1

with open(OUT, "w", encoding="utf-8") as f:
    for r in ids:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

cats = {}
for r in ids:
    cats[r["category"]] = cats.get(r["category"], 0) + 1
print(f"wrote {out_n} prompts to {OUT}")
print("categories:", json.dumps(cats, ensure_ascii=False))
langs = {}
for r in ids:
    langs[r["expect_lang"]] = langs.get(r["expect_lang"], 0) + 1
print("expect_lang:", langs)
assert out_n >= 200, f"need >= 200 prompts, got {out_n}"
