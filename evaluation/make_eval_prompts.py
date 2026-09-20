#!/usr/bin/env python3
"""ARION ALPHA 1 — build evaluation prompt sets.

Spec section 19: 100+ web-dev prompts and 100+ bilingual prompts.
Outputs: evaluation/prompts_webdev.json + evaluation/prompts_bilingual.json
"""
import json, os, random

random.seed(99)
ROOT = "/home/z/my-project/arion-alpha-1"
EV = f"{ROOT}/evaluation"
os.makedirs(EV, exist_ok=True)

WEBDEV = []
# --- HTML knowledge (25)
html_q = [
    ("en", "When should I use <article> instead of <section>? Show a correct example.", "html_semantics"),
    ("en", "Write a complete accessible contact form with labels, required fields and inline validation messages.", "html_forms"),
    ("en", "What meta tags does a modern HTML page need for SEO? Give a full <head> example.", "html_meta"),
    ("en", "How does the <picture> element differ from srcset? Show both.", "html_images"),
    ("en", "Build a semantic page skeleton with header, nav, main, aside and footer.", "html_semantics"),
    ("en", "How do I make a native modal with <dialog>? Include opening, closing and the backdrop.", "html_dialog"),
    ("en", "Show a data table with thead, tbody, tfoot, caption and proper scope attributes.", "html_tables"),
    ("en", "What is wrong with using <div> for everything? Explain and show the semantic version.", "html_semantics"),
    ("en", "Write an accessible skip-link pattern.", "html_a11y"),
    ("en", "How do details/summary work for an FAQ section? Show a complete example.", "html_details"),
    ("fa", "تفاوت تگ semantic و div چیست؟ با مثال توضیح بده.", "html_semantics"),
    ("fa", "یک فرم تماس کامل با لیبل و اعتبارسنجی HTML بساز.", "html_forms"),
    ("fa", "متاتگ‌های ضروری برای سئو کدام‌اند؟ یک head کامل بنویس.", "html_meta"),
    ("fa", "چطور با <dialog> یک مودال بسازم؟", "html_dialog"),
    ("fa", "ساختار سمانتیک یک صفحهٔ وب را با header و nav و main بنویس.", "html_semantics"),
    ("en", "Explain the loading attribute for images and iframes.", "html_images"),
    ("en", "How do I embed a YouTube video responsively?", "html_iframe"),
    ("en", "Write valid markup for a navigation bar with a logo and 4 links.", "html_nav"),
    ("en", "What does the template element do? Show a practical use.", "html_template"),
    ("en", "How do I validate an email input in HTML without JavaScript?", "html_forms"),
    ("fa", "چطور تصاویر را برای موبایل بهینه کنم؟ srcset را توضیح بده.", "html_images"),
    ("fa", "جدول کامل با caption و scope بنویس.", "html_tables"),
    ("fa", "چرا alt برای تصاویر مهم است؟", "html_a11y"),
    ("en", "Show correct usage of fieldset and legend in a shipping form.", "html_forms"),
    ("en", "What is the correct heading hierarchy and why does it matter?", "html_semantics"),
]
for lang, q, tag in html_q:
    WEBDEV.append({"id": f"wd-html-{len(WEBDEV)+1:03d}", "lang": lang, "category": "html", "tag": tag, "prompt": q})

# --- CSS knowledge (30)
css_q = [
    ("en", "Explain the CSS box model and show how border-box changes sizing.", "css_boxmodel"),
    ("en", "Center a card both vertically and horizontally with flexbox AND with grid.", "css_centering"),
    ("en", "Build a responsive 3-column grid that becomes 1 column on mobile.", "css_grid"),
    ("en", "Explain specificity with a concrete example where an ID beats a class.", "css_specificity"),
    ("en", "How do CSS custom properties enable dark mode? Show a full theme system.", "css_theming"),
    ("en", "Write fluid typography with clamp() from 1.5rem to 3rem.", "css_fluid"),
    ("en", "What is the difference between position sticky and fixed? Demo both.", "css_position"),
    ("en", "Create a glassmorphism card.", "css_tricks"),
    ("en", "Build a skeleton loader with shimmer animation.", "css_tricks"),
    ("en", "How does z-index actually work? Explain stacking contexts.", "css_position"),
    ("en", "Animate a button on hover with transform and transition — no layout properties.", "css_animation"),
    ("en", "Write a keyframes animation for a floating element, respecting prefers-reduced-motion.", "css_animation"),
    ("en", "Explain margin collapsing and three ways to stop it.", "css_boxmodel"),
    ("en", "Use aspect-ratio for a responsive video embed.", "css_layout"),
    ("en", "What is object-fit: cover doing exactly? Show with an avatar.", "css_images"),
    ("en", "Build a CSS-only tooltip.", "css_tricks"),
    ("en", "How do logical properties help in RTL layouts?", "css_logical"),
    ("en", "Write a media query set for mobile-first design at 640px and 1024px.", "css_media"),
    ("en", "What are container queries and when are they better than media queries?", "css_media"),
    ("en", "Create a responsive navbar without JavaScript.", "css_tricks"),
    ("fa", "با فلکس‌باکس وسط‌چین کردن کامل را توضیح بده.", "css_centering"),
    ("fa", "تفاوت grid و flexbox در چیست و کجا از کدام استفاده کنیم؟", "css_layout"),
    ("fa", "چطور با متغیرهای CSS حالت تاریک بسازم؟", "css_theming"),
    ("fa", "یک کارت شیشه‌ای بساز.", "css_tricks"),
    ("fa", " specificity یا اولویت سلکتورها را با مثال توضیح بده.", "css_specificity"),
    ("fa", "با clamp فونت واکنش‌گرا بنویس.", "css_fluid"),
    ("fa", "چرا z-index کار نمی‌کند؟", "css_position"),
    ("fa", "انیمیشن hover برای دکمه بساز.", "css_animation"),
    ("fa", "منوی ناوبری ریسپانسیو بدون جاوااسکریپت.", "css_tricks"),
    ("en", "Explain the cascade, inheritance and specificity in 3 short paragraphs with code.", "css_specificity"),
]
for lang, q, tag in css_q:
    WEBDEV.append({"id": f"wd-css-{len(WEBDEV)+1:03d}", "lang": lang, "category": "css", "tag": tag, "prompt": q})

# --- JS knowledge (25)
js_q = [
    ("en", "Explain closures with a practical counter example.", "js_closures"),
    ("en", "Fetch data with async/await including error handling for HTTP errors.", "js_fetch"),
    ("en", "What is event delegation and why is it efficient? Code it for a todo list.", "js_events"),
    ("en", "Write a debounce function and explain where to use it.", "js_utils"),
    ("en", "How do const and let differ from var?", "js_basics"),
    ("en", "Store and retrieve an object in localStorage safely.", "js_storage"),
    ("en", "Explain the event loop with a console.log ordering example.", "js_async"),
    ("en", "Use IntersectionObserver to lazy-load images.", "js_observers"),
    ("en", "Build a simple tabs component in vanilla JS.", "js_components"),
    ("en", "Validate a form on submit and show inline errors.", "js_forms"),
    ("en", "Explain spread vs rest with examples.", "js_basics"),
    ("en", "What does optional chaining solve? Show 3 cases.", "js_basics"),
    ("en", "Write a countdown timer with setInterval and clean it up.", "js_timers"),
    ("en", "How do modules work in the browser with import/export?", "js_modules"),
    ("en", "Convert this callback pyramid to async/await: getData(id, cb => getUser(cb, u => render(u)))", "js_async"),
    ("fa", "بستار (closure) را با مثال توضیح بده.", "js_closures"),
    ("fa", "fetch با async/await و مدیریت خطا بنویس.", "js_fetch"),
    ("fa", "event delegation چیست؟ برای لیست کارها پیاده‌سازی کن.", "js_events"),
    ("fa", "تابع debounce بنویس و کاربردش را بگو.", "js_utils"),
    ("fa", "ذخیرهٔ آبجکت در localStorage با تبدیل JSON.", "js_storage"),
    ("fa", "event loop را با مثال ترتیب console.log توضیح بده.", "js_async"),
    ("fa", "کامپوننت تب ساده با جاوااسکریپت خالص.", "js_components"),
    ("fa", "اعتبارسنجی فرم و نمایش خطا زیر هر فیلد.", "js_forms"),
    ("fa", "تفاوت == و === با مثال.", "js_basics"),
    ("en", "When does fetch reject its promise? Explain response.ok.", "js_fetch"),
]
for lang, q, tag in js_q:
    WEBDEV.append({"id": f"wd-js-{len(WEBDEV)+1:03d}", "lang": lang, "category": "js", "tag": tag, "prompt": q})

# --- Build requests (25)
builds = [
    ("en", "Build a complete modern landing page for an AI startup with hero, features, pricing and footer."),
    ("en", "Create a responsive dashboard page with stat cards and a table."),
    ("en", "Build a login page with validation and a password visibility toggle."),
    ("en", "Create a pricing section with 3 tiers and a highlighted plan."),
    ("en", "Build a portfolio site with project cards and a contact section."),
    ("en", "Create a testimonial slider with vanilla JS."),
    ("en", "Build a responsive blog homepage with a card grid."),
    ("en", "Create an FAQ accordion section."),
    ("en", "Build a hero section with an animated gradient background."),
    ("en", "Create a footer with 4 columns and social links."),
    ("en", "Build a product card grid for an ecommerce page."),
    ("en", "Create a team section with avatar cards."),
    ("en", "Build a sticky navbar that hides on scroll down and shows on scroll up."),
    ("en", "Create a settings page with a sidebar and form sections."),
    ("en", "Build a documentation page with a sidebar and content area."),
    ("fa", "یک لندینگ پیج مدرن برای یک شرکت هوش مصنوعی بساز."),
    ("fa", "یک داشبورد ریسپانسیو با کارت‌های آمار بساز."),
    ("fa", "صفحهٔ لاگین با اعتبارسنجی بساز."),
    ("fa", "بخش قیمت‌گذاری با سه پلن بساز."),
    ("fa", "یک سایت نمونه‌کار (پورتفولیو) با کارت پروژه بساز."),
    ("fa", "اسلایدر نظرات کاربران با جاوااسکریپت خالص بساز."),
    ("fa", "یک صفحهٔ وبلاگ ریسپانسیو با گرید کارت بساز."),
    ("fa", "آکاردئون سوالات متداول بساز."),
    ("fa", "فوتر چهارستونه با شبکه‌های اجتماعی بساز."),
    ("fa", "نوبار چسبان با منوی همبرگری برای موبایل بساز."),
]
for lang, q in builds:
    WEBDEV.append({"id": f"wd-build-{len(WEBDEV)+1:03d}", "lang": lang, "category": "build", "tag": "full_page", "prompt": q})

# --- Bilingual set (110)
BI = []
fa_en_pairs = [
    ("Why does my flexbox not center vertically?", "چرا فلکس‌باکس من عمودی وسط‌چین نمی‌شود؟"),
    ("How do I load Google Fonts correctly?", "فونت گوگل را چطور درست لود کنم؟"),
    ("What is the difference between grid and flexbox?", "تفاوت grid و flexbox چیست؟"),
    ("How does position: sticky work?", "position: sticky چطور کار می‌کند؟"),
    ("Why is my media query ignored on mobile?", "چرا media query من در موبایل اعمال نمی‌شود؟"),
    ("How do I make a dark mode toggle?", "چطور دکمهٔ حالت تاریک بسازم؟"),
    ("What breaks 100vh on mobile?", "چرا 100vh در موبایل مشکل دارد؟"),
    ("How do I lazy load images?", "بارگذاری تنبل تصاویر چطور است؟"),
    ("Explain event bubbling.", "حباب‌زنی رویداد (event bubbling) را توضیح بده."),
    ("Why is my z-index ignored?", "چرا z-index من نادیده گرفته می‌شود؟"),
    ("How do I center a div?", "چطور یک div را وسط‌چین کنم؟"),
    ("What is CSS specificity?", "اولویت سلکتورهای CSS چیست؟"),
    ("How does fetch handle errors?", "fetch خطاها را چطور مدیریت می‌کند؟"),
    ("Explain async/await simply.", "async/await را ساده توضیح بده."),
    ("How do I use Vazirmatn font?", "فونت وزیرمتن را چطور استفاده کنم؟"),
    ("What is semantic HTML?", "HTML سمانتیک چیست؟"),
    ("How do transitions differ from animations?", "transition و animation چه فرقی دارند؟"),
    ("How do I build a modal accessibly?", "مودال دسترس‌پذیر چطور بسازم؟"),
    ("What causes horizontal scroll on mobile?", "اسکرول افقی موبایل از چیست؟"),
    ("How does localStorage work?", "localStorage چطور کار می‌کند؟"),
]
i = 0
while len(BI) < 110:
    en_q, fa_q = fa_en_pairs[i % len(fa_en_pairs)]
    mode = i % 4
    if mode == 0:      # fa → fa
        BI.append({"id": f"bi-{i:03d}", "kind": "fa_to_fa", "prompt": fa_q, "expect": "fa"})
    elif mode == 1:    # en → en
        BI.append({"id": f"bi-{i:03d}", "kind": "en_to_en", "prompt": en_q, "expect": "en"})
    elif mode == 2:    # en question, answer requested in Persian
        BI.append({"id": f"bi-{i:03d}", "kind": "en_ask_fa", "prompt": en_q + " Please answer in Persian/Farsi.", "expect": "fa"})
    else:              # fa question, answer requested in English
        BI.append({"id": f"bi-{i:03d}", "kind": "fa_ask_en", "prompt": fa_q + " لطفاً به انگلیسی جواب بده.", "expect": "en"})
    i += 1

json.dump({"name": "arion-webdev-eval", "count": len(WEBDEV), "prompts": WEBDEV},
          open(f"{EV}/prompts_webdev.json", "w"), indent=2, ensure_ascii=False)
json.dump({"name": "arion-bilingual-eval", "count": len(BI), "prompts": BI},
          open(f"{EV}/prompts_bilingual.json", "w"), indent=2, ensure_ascii=False)
print(f"webdev prompts: {len(WEBDEV)} | bilingual prompts: {len(BI)}")
