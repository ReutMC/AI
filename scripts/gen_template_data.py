#!/usr/bin/env python3
"""ARION ALPHA 1 — template-based synthetic dataset generator.

Produces hundreds of genuinely-varied, complete, working web-dev training
examples (randomized palettes, fonts, copy, section orders) in JSONL chat format.
Deterministic via fixed seed.
"""
import json, random, os

random.seed(4217)
OUT = "/home/z/my-project/arion-alpha-1/datasets/generated/template_data.jsonl"

SYS = ("You are Arion Alpha 1, a professional bilingual (Persian/English) web development AI "
       "specialized in HTML, CSS, JavaScript and modern frontend development. Answer in the "
       "same language as the user, with complete working code and clear explanations.")

rows = []

def add(cat, lang, diff, user, asst, system=True, cid=None):
    msgs = ([{"role": "system", "content": SYS}] if system else []) + [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": asst.strip()},
    ]
    rows.append({
        "id": cid or f"tpl-{len(rows):05d}",
        "category": cat, "lang": lang, "difficulty": diff, "messages": msgs,
    })

# ---------------------------------------------------------------- palettes/fonts
PALETTES = [
    ("Midnight Emerald", "#0f172a", "#10b981", "#f8fafc", "#1e293b", "#34d399"),
    ("Warm Sunset", "#1c1917", "#f97316", "#fffbeb", "#292524", "#fb923c"),
    ("Royal Violet", "#18181b", "#8b5cf6", "#fafafa", "#27272a", "#a78bfa"),
    ("Ocean Slate", "#0c1420", "#06b6d4", "#f0f9ff", "#16283a", "#22d3ee"),
    ("Rosewood", "#1a1215", "#e11d48", "#fff1f2", "#2a1a20", "#fb7185"),
    ("Forest Paper", "#14201a", "#22c55e", "#f7fee7", "#1d2f26", "#4ade80"),
    ("Amber Dusk", "#231a10", "#d97706", "#fffbeb", "#332617", "#f59e0b"),
    ("Graphite Red", "#171717", "#ef4444", "#f5f5f5", "#262626", "#f87171"),
]
EN_FONTS = [
    ("Inter", "Inter:wght@400;600;800"), ("Manrope", "Manrope:wght@400;600;800"),
    ("Space Grotesk", "Space+Grotesk:wght@400;600;700"), ("Sora", "Sora:wght@400;600;800"),
    ("Outfit", "Outfit:wght@400;600;800"), ("Urbanist", "Urbanist:wght@400;600;800"),
]
FA_FONTS = [("Vazirmatn", "Vazirmatn:wght@400;600;800"), ("Estedad", "Estedad:wght@400;600;800")]

BRANDS_EN = ["Nova", "Lumina", "Vertex", "Solstice", "Nimbus", "Orbitly", "Craftline", "Brightpath", "Skyforge", "Pixelport"]
BRANDS_FA = ["آریا‌تک", "نگین‌وب", "رهیاب", "سپهرنو", "کیان‌دیجیتال", "تابان‌مدیا", "فرازنت", "داده‌گستر"]

FA_WORDS = {
    "hero_t": ["راه‌حل هوشمند برای کسب‌وکار شما", "وب‌سایت شما، ویترین حرفه‌ای شما", "آیندهٔ دیجیتال را با هم بسازیم", "تجربه‌ای متفاوت از وب"],
    "hero_s": ["ابزارهای مدرن، طراحی واکنش‌گرا و عملکرد سریع در یک پکیج کامل.", "از ایده تا اجرا، همراه شما هستیم.", "طراحی زیبا، کد تمیز و تجربهٔ کاربری بی‌نقص."],
    "cta": ["شروع کنید", "همین حالا بسازید", "مشاهدهٔ دمو", "درخواست دمو"],
    "feat_t": ["سرعت بالا", "طراحی واکنش‌گرا", "پشتیبانی فارسی", "امنیت داده", "شخصی‌سازی کامل", "بهینه‌سازی سئو"],
    "feat_s": ["بارگذاری صفحه در کمتر از یک ثانیه.", "نمایش بی‌نقص در موبایل، تبلت و دسکتاپ.", "تیم پشتیبانی همیشه پاسخ‌گو است.", "داده‌های شما با رمزنگاری محافظت می‌شود.", "قالب‌ها را مطابق سلیقهٔ خود تغییر دهید.", "ساختار استاندارد برای موتورهای جست‌وجو."],
    "nav": ["خانه", "امکانات", "قیمت‌گذاری", "بلاگ", "تماس", "دربارهٔ ما", "خدمات"],
}
EN_WORDS = {
    "hero_t": ["Ship your next idea faster", "Beautiful websites, built right", "The platform for modern teams", "Design once, deploy everywhere"],
    "hero_s": ["Everything you need to launch a polished product — no bloat, just speed.", "Clean code, responsive layouts and delightful interactions by default.", "From prototype to production in a single afternoon."],
    "cta": ["Get started", "Try it free", "See demo", "Book a call"],
    "feat_t": ["Blazing fast", "Fully responsive", "Accessible by default", "SEO friendly", "Dark mode", "Zero config"],
    "feat_s": ["Pages load in under a second on slow networks.", "Looks perfect on phones, tablets and desktops.", "Semantic markup and focus states out of the box.", "Clean structure that search engines understand.", "One toggle switches the whole palette.", "No build step required — just open and edit."],
    "nav": ["Home", "Features", "Pricing", "Blog", "Contact", "About", "Services"],
}

def fonts_link(fa=False):
    fam = random.choice(FA_FONTS if fa else EN_FONTS)
    return fam[0], (f'<link rel="preconnect" href="https://fonts.googleapis.com">\n'
                    f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
                    f'<link href="https://fonts.googleapis.com/css2?family={fam[1]}&display=swap" rel="stylesheet">')

def page(name, fa, style, body, script=""):
    f0, flink = fonts_link(fa)
    w = FA_WORDS if fa else EN_WORDS
    langdir = 'lang="fa" dir="rtl"' if fa else 'lang="en"'
    fam_css = "Vazirmatn" if fa else name
    title = f"{random.choice(BRANDS_FA if fa else BRANDS_EN)} — {w['hero_t'][0] if fa else 'Modern site'}"
    return f"""<!DOCTYPE html>
<html {langdir}>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {flink}
  <style>
{style}
  </style>
</head>
<body>
{body}
{script}
</body>
</html>"""

def nav_html(fa, brand):
    w = FA_WORDS if fa else EN_WORDS
    picks = random.sample(w["nav"], 4)
    links = "".join(f'\n      <a href="#{i}">{t}</a>' for t in picks)
    return f'''  <header class="navbar">
    <a class="brand" href="#">{brand}</a>
    <button class="menu-btn" aria-label="{'منو' if fa else 'Menu'}" aria-expanded="false">☰</button>
    <nav class="nav-links">{links}
    </nav>
  </header>'''

def hero_html(fa):
    w = FA_WORDS if fa else EN_WORDS
    t, s, c = random.choice(w["hero_t"]), random.choice(w["hero_s"]), random.choice(w["cta"])
    return f'''  <section class="hero">
    <h1>{t}</h1>
    <p>{s}</p>
    <a class="btn" href="#features">{c}</a>
  </section>'''

def feats_html(fa, n=3):
    w = FA_WORDS if fa else EN_WORDS
    idx = random.sample(range(len(w["feat_t"])), n)
    cards = "".join(
        f'''\n    <article class="card">
      <h3>{w["feat_t"][i]}</h3>
      <p>{w["feat_s"][i]}</p>
    </article>''' for i in idx)
    return f'  <section id="features" class="features">{cards}\n  </section>'

BASE_CSS = """    :root {{
      --bg: {bg}; --surface: {surface}; --text: {txt}; --accent: {acc}; --accent-2: {acc2};
      --radius: 14px;
    }}
    * {{ box-sizing: border-box; margin: 0; }}
    body {{ font-family: '{font}', system-ui, {fa_stack}; background: var(--bg); color: var(--text); line-height: 1.6; }}
    .navbar {{ display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.5rem; position: sticky; top: 0; background: color-mix(in srgb, var(--bg) 85%, transparent); backdrop-filter: blur(10px); z-index: 10; }}
    .brand {{ font-weight: 800; font-size: 1.25rem; color: var(--text); text-decoration: none; }}
    .nav-links a {{ color: var(--text); text-decoration: none; margin-inline-start: 1.25rem; opacity: .8; transition: opacity .2s; }}
    .nav-links a:hover {{ opacity: 1; color: var(--accent); }}
    .menu-btn {{ display: none; background: none; border: 1px solid var(--surface); color: var(--text); font-size: 1.25rem; padding: .35rem .7rem; border-radius: 8px; cursor: pointer; }}
    .hero {{ min-height: 70vh; display: grid; place-content: center; text-align: center; padding: 2rem; gap: 1rem; }}
    .hero h1 {{ font-size: clamp(2rem, 5vw + 1rem, 4rem); line-height: 1.15; letter-spacing: -0.02em; }}
    .hero p {{ max-width: 55ch; opacity: .75; margin-inline: auto; }}
    .btn {{ justify-self: center; background: var(--accent); color: var(--bg); padding: .8rem 1.6rem; border-radius: 999px; text-decoration: none; font-weight: 700; transition: transform .2s, box-shadow .2s; }}
    .btn:hover {{ transform: translateY(-2px); box-shadow: 0 10px 30px color-mix(in srgb, var(--accent) 40%, transparent); }}
    .features {{ display: grid; gap: 1.25rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); padding: 4rem 1.5rem; max-width: 1100px; margin-inline: auto; }}
    .card {{ background: var(--surface); border: 1px solid color-mix(in srgb, var(--text) 10%, transparent); border-radius: var(--radius); padding: 1.5rem; transition: transform .25s; }}
    .card:hover {{ transform: translateY(-4px); }}
    .card h3 {{ color: var(--accent); margin-bottom: .5rem; }}
    .card p {{ opacity: .7; font-size: .95rem; }}
    @media (max-width: 720px) {{
      .menu-btn {{ display: block; }}
      .nav-links {{ position: fixed; inset: 60px 0 auto 0; background: var(--surface); flex-direction: column; padding: 1rem 1.5rem; display: none; border-radius: 0 0 var(--radius) var(--radius); }}
      .nav-links.open {{ display: flex; }}
      .nav-links a {{ display: block; margin: .5rem 0; }}
    }}"""

NAV_JS = """  <script>
    const btn = document.querySelector('.menu-btn');
    const links = document.querySelector('.nav-links');
    btn.addEventListener('click', () => {
      const open = links.classList.toggle('open');
      btn.setAttribute('aria-expanded', open);
    });
  </script>"""

def build_example(fa, with_js=True, cid=None):
    pal = random.choice(PALETTES)
    f0, _ = fonts_link(fa)
    brand = random.choice(BRANDS_FA if fa else BRANDS_EN)
    style = BASE_CSS.format(bg=pal[1], surface=pal[4], txt=pal[3], acc=pal[2], acc2=pal[5],
                            font=f0, fa_stack="'Vazirmatn', Tahoma, sans-serif" if fa else "sans-serif")
    body = nav_html(fa, brand) + "\n" + hero_html(fa) + "\n" + feats_html(fa, random.choice([3, 4, 6]))
    html = page(f0, fa, style, body, NAV_JS if with_js else "")
    w = "fa" if fa else "en"
    if fa:
        u = f"یک {random.choice(['لندینگ پیج مدرن', 'صفحهٔ فرود ساده', 'سایت معرفی شرکت'])} برای «{brand}» بساز؛ رنگ اصلی {pal[2]}، فونت وزیرمتن، کاملاً ریسپانسیو و RTL."
        a = (f"این یک صفحهٔ کامل و قابل‌اجرا است؛ فایل را با نام `index.html` ذخیره کنید:\n\n```html\n{html}\n```\n\n"
             "**چرا این کد کار می‌کند:**\n"
             "- `dir=\"rtl\"` روی تگ `<html>` کل چیدمان را راست‌به‌چپ می‌کند و ویژگی‌های منطقی مثل `margin-inline-start` جهت را خودکار جابه‌جا می‌کنند.\n"
             f"- فونت «{f0}» با دو `preconnect` زودتر از `stylesheet` لود می‌شود تا تأخیر فونت کم شود و `display=swap` از متن نامرئی جلوگیری می‌کند.\n"
             "- `clamp()` اندازهٔ تیتر را بین دو حد با نرخ واکنش‌گرا نگه می‌دارد، بدون media query.\n"
             "- منوی موبایل فقط با `classList.toggle` باز و بسته می‌شود و `aria-expanded` برای صفحه‌خوان‌ها به‌روز می‌ماند.")
        add("fa_build", "fa", "intermediate", u, a, cid=cid)
    else:
        u = random.choice([
            f"Build a modern landing page for a startup called {brand}. Accent color {pal[2]}, mobile-first, sticky navbar.",
            f"Create a complete single-file landing page for {brand} with a features grid and a working mobile menu.",
            f"I need a clean hero + features section for {brand}. Use the {pal[0]} palette with {pal[2]} as the primary color.",
        ])
        a = (f"Here is a complete, working single-file page — save it as `index.html`:\n\n```html\n{html}\n```\n\n"
             "**Why this works:**\n"
             "- `place-content: center` on the hero grid vertically and horizontally centers the content without absolute positioning.\n"
             "- `clamp()` makes the headline fluid: it grows with the viewport between the min and max bounds, so no extra media query is needed.\n"
             "- The navbar uses `backdrop-filter: blur()` over a semi-transparent background (via `color-mix`) for a glass effect while staying readable.\n"
             "- The mobile menu is progressive enhancement: it is hidden by CSS below 720px and toggled with `aria-expanded` kept in sync for screen readers.\n"
             "- Cards use `auto-fit` + `minmax(240px, 1fr)` so the grid reflows from 1 to 3 columns automatically.")
        add("en_build", "en", "intermediate", u, a, cid=cid)

for i in range(46):
    build_example(True, cid=f"tpl-fabuild-{i:03d}")
for i in range(70):
    build_example(False, cid=f"tpl-enbuild-{i:03d}")

# ---------------------------------------------------------------- CSS trick cards
def glass_card(en=True, cid=None):
    pal = random.choice(PALETTES)
    css = f'''.glass {{
  background: color-mix(in srgb, {pal[4]} 55%, transparent);
  backdrop-filter: blur(16px) saturate(160%);
  border: 1px solid color-mix(in srgb, {pal[3]} 18%, transparent);
  border-radius: 18px;
  padding: 2rem;
  box-shadow: 0 8px 32px rgb(0 0 0 / 0.35);
}}'''
    html = f'''<div class="glass">
  <h2>{'کارت شیشه‌ای' if not en else 'Glassmorphism card'}</h2>
  <p>{'محتوای شما روی پس‌زمینهٔ تار و شفاف.' if not en else 'Your content floats over a frosted backdrop.'}</p>
</div>'''
    if en:
        add("pro_csstricks", "en", "advanced", "How do I make a glassmorphism card in CSS?",
            f"```html\n{html}\n```\n\n```css\n{css}\n```\n\n**Why it works:** `backdrop-filter: blur()` blurs whatever is painted *behind* the element — the card itself stays semi-transparent through `color-mix(... transparent)`, so the blurred backdrop shows through. `saturate(160%)` keeps colors vivid behind the frost, and the 1px translucent border simulates the light edge real glass catches.", cid=cid)
    else:
        add("pro_csstricks", "fa", "advanced", "چطور در CSS یک کارت شیشه‌ای (glassmorphism) بسازم؟",
            f"```html\n{html}\n```\n\n```css\n{css}\n```\n\n**چرا کار می‌کند:** `backdrop-filter: blur()` هر چیزی که *پشت* عنصر است را تار می‌کند؛ خود کارت نیمه‌شفاف می‌ماند تا پس‌زمینهٔ تار دیده شود. `saturate` رنگ‌های پشت شیشه را زنده نگه می‌دارد و حاشیهٔ ۱ پیکسلی نیمه‌شفاف، حالت لبهٔ نورگیر شیشه را شبیه‌سازی می‌کند.", cid=cid)

def skeleton_loader(en=True, cid=None):
    css = '''.skeleton {
  background: linear-gradient(90deg, #e5e7eb 25%, #f3f4f6 50%, #e5e7eb 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite linear;
  border-radius: 8px;
}
@keyframes shimmer {
  from { background-position: 200% 0; }
  to   { background-position: -200% 0; }
}
@media (prefers-reduced-motion: reduce) { .skeleton { animation: none; } }'''
    html = '<div class="skeleton" style="width: 240px; height: 16px;"></div>\n<div class="skeleton" style="width: 180px; height: 16px; margin-top: 8px;"></div>'
    u_en = "Show me how to build a skeleton loading placeholder in pure CSS."
    a_en_body = f"```html\n{html}\n```\n\n```css\n{css}\n```\n\n**Why it works:** the shimmer is a linear-gradient wider than the element (`background-size: 200%`) whose `background-position` animates from one side to the other — no extra DOM. `prefers-reduced-motion` disables the loop for users who opt out of motion."
    u_fa = "یک placeholder اسکلتی (skeleton) فقط با CSS بساز"
    a_fa = f"```html\n{html}\n```\n\n```css\n{css}\n```\n\n**چرا کار می‌کند:** گرادیان خطی از عرضِ خود عنصر بزرگ‌تر است (`background-size: 200%`) و انیمیشنِ `background-position` آن را به‌صورت موج به حرکت درمی‌آورد؛ بدون هیچ DOM اضافه. با `prefers-reduced-motion` انیمیشن برای کاربرانی که حرکت را محدود کرده‌اند خاموش می‌شود."
    if en: add("pro_csstricks", "en", "intermediate", u_en, a_en_body, cid=cid)
    else:  add("pro_csstricks", "fa", "intermediate", u_fa, a_fa, cid=cid)

def theme_switcher(cid=None):
    css = ''':root {
  --bg: #0b0f14; --text: #e6edf3; --surface: #151b23; --accent: #2dd4bf;
}
[data-theme="light"] {
  --bg: #f8fafc; --text: #0f172a; --surface: #ffffff; --accent: #0d9488;
}
body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; transition: background .3s, color .3s; }
button { background: var(--accent); color: var(--bg); border: 0; padding: .6rem 1.2rem; border-radius: 8px; cursor: pointer; }'''
    js = '''const btn = document.querySelector('#theme');
const saved = localStorage.getItem('theme');
if (saved) document.documentElement.dataset.theme = saved;
btn.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('theme', next);
});'''
    html = '<button id="theme">Toggle theme</button>\n<main><h1>Themed page</h1><p>Colors come from custom properties.</p></main>'
    add("pro_jspatterns", "en", "intermediate",
        "Build a dark/light theme toggle that remembers the choice across reloads.",
        f"```html\n{html}\n```\n\n```css\n{css}\n```\n\n```js\n{js}\n```\n\n**Why it works:** all colors flow from custom properties, so switching themes only swaps the variable values on `:root` via the `data-theme` attribute — no restyling of individual elements. The choice is persisted in `localStorage` and applied *before* first paint (script runs on load), avoiding a flash of the wrong theme.", cid=cid)

def scroll_reveal(cid=None):
    js = '''const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      io.unobserve(e.target); // reveal once
    }
  }
}, { threshold: 0.15 });

document.querySelectorAll('.reveal').forEach(el => io.observe(el));'''
    css = '''.reveal { opacity: 0; transform: translateY(24px); transition: opacity .6s ease, transform .6s ease; }
.reveal.visible { opacity: 1; transform: none; }
@media (prefers-reduced-motion: reduce) { .reveal { opacity: 1; transform: none; transition: none; } }'''
    html = '<section class="reveal">Card one</section>\n<section class="reveal">Card two</section>\n<section class="reveal">Card three</section>'
    add("pro_jspatterns", "en", "intermediate",
        "Add a scroll-reveal animation to sections without any library.",
        f"```html\n{html}\n```\n\n```css\n{css}\n```\n\n```js\n{js}\n```\n\n**Why it works:** `IntersectionObserver` fires off the main thread whenever an element's visibility crosses the threshold — far cheaper than scroll listeners computing `getBoundingClientRect()` on every frame. Elements start invisible via CSS, get the `.visible` class once seen, and are unobserved to free memory. The reduced-motion query respects user preferences.", cid=cid)

def tooltip_css(cid=None):
    css = '''[data-tip] { position: relative; }
[data-tip]::after {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 8px);
  left: 50%;
  translate: -50% 4px;
  background: #111827; color: #f9fafb;
  padding: .35rem .6rem; border-radius: 6px; font-size: .8rem;
  white-space: nowrap;
  opacity: 0; pointer-events: none;
  transition: opacity .2s, translate .2s;
}
[data-tip]:hover::after, [data-tip]:focus-visible::after { opacity: 1; translate: -50% 0; }'''
    html = '<button data-tip="Copy to clipboard">⧉ Copy</button>'
    add("pro_csstricks", "en", "intermediate", "Can I make a tooltip with CSS only, no JS?",
        f"```html\n{html}\n```\n\n```css\n{css}\n```\n\n**Why it works:** the tooltip text lives in a `data-*` attribute and is rendered by the `::after` pseudo-element through `content: attr(data-tip)` — no extra markup. It is positioned above the button, hidden with `opacity: 0` + `pointer-events: none`, and fades in on `:hover` **and** `:focus-visible` so keyboard users get it too.", cid=cid)

for i in range(6):  glass_card(True, cid=f"tpl-glass-en-{i}")
for i in range(4):  glass_card(False, cid=f"tpl-glass-fa-{i}")
for i in range(3):  skeleton_loader(True, cid=f"tpl-skel-en-{i}")
skeleton_loader(False, cid="tpl-skel-fa-0")
for i in range(2):  tooltip_css(cid=f"tpl-tip-{i}")
scroll_reveal(cid="tpl-reveal-0")
theme_switcher(cid="tpl-theme-0")

# ---------------------------------------------------------------- explanation facts
CSS_FACTS = [
 ("z-index", "z-index only works on positioned elements (position other than static) or flex/grid children. If it seems ignored, the element is probably not positioned, or you are fighting a different stacking context created by an ancestor with transform/opacity/filter.",
  '.box {\n  position: relative; /* required */\n  z-index: 10;\n}'),
 ("specificity", "Specificity is a three-column score: (inline, id, class/attr/pseudo-class, element). When two rules target the same element, the higher score wins regardless of file order; equal scores fall back to source order. !important overrides everything and should be a last resort.",
  '/* id (1-0-0) beats class (0-1-0) */\n#hero .title { color: red; }\n.title { color: blue; } /* loses */'),
 ("box-sizing", "By default, width/height apply to the content box only, so padding and border inflate your layout. border-box makes the declared width include padding and border, which is why nearly every project starts with * { box-sizing: border-box; }.",
  '*, *::before, *::after { box-sizing: border-box; }\n.card { width: 300px; padding: 1rem; /* still 300px wide */ }'),
 ("position: sticky", "sticky elements behave relative until a threshold is crossed, then stay fixed within their nearest scrolling ancestor — but they can never leave their parent. If your sticky header 'disappears', its parent is probably shorter than the scroll area or an ancestor has overflow: hidden.",
  '.header {\n  position: sticky;\n  top: 0;\n  z-index: 5;\n}'),
 ("flexbox centering", "On a flex container, justify-content aligns items on the main axis and align-items on the cross axis. To center both ways: display:flex; justify-content:center; align-items:center; (or the grid one-liner: display:grid; place-items:center).",
  '.center {\n  display: flex;\n  justify-content: center; /* main axis */\n  align-items: center;       /* cross axis */\n  min-height: 100vh;\n}'),
 ("CSS Grid auto-fit", "repeat(auto-fit, minmax(240px, 1fr)) creates as many ≥240px columns as fit, letting each grow equally. auto-fit collapses empty tracks, so with one card it stretches full width; auto-fill keeps empty tracks reserved.",
  '.grid {\n  display: grid;\n  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));\n  gap: 1rem;\n}'),
 ("clamp()", "clamp(min, preferred, max) resolves the middle value while clamping the result between the bounds. With viewport units in the middle it produces fluid typography that never gets too small or too large — replacing several media queries.",
  'h1 {\n  font-size: clamp(1.8rem, 4vw + 1rem, 3.5rem);\n}'),
 ("CSS custom properties", "Custom properties (--name) participate in the cascade and inherit, unlike Sass variables which compile away. Redefining them on a scope (like [data-theme='light']) instantly re-themes everything that uses them, and they can change at runtime from JS.",
  ':root { --accent: #10b981; }\n.btn { background: var(--accent, #333); }'),
 ("aspect-ratio", "aspect-ratio declares the preferred width/height ratio. It is used whenever one dimension is auto, so width: 100%; aspect-ratio: 16 / 9 keeps videos and cards perfectly proportional without padding-top hacks.",
  '.video { width: 100%; aspect-ratio: 16 / 9; object-fit: cover; }'),
 ("object-fit", "object-fit controls how an img/video fills its box when aspect ratios differ: cover scales and crops to fill (no distortion), contain fits entirely with letterboxing, fill stretches (usually unwanted).",
  'img.avatar { width: 64px; height: 64px; border-radius: 50%; object-fit: cover; }'),
 ("pseudo-elements", "::before and ::after insert a child box at the start/end of the element's content. They require the content property (even empty ''), can be styled freely, and are perfect for decorations that don't belong in the DOM.",
  '.badge::after {\n  content: "";\n  display: inline-block;\n  width: 8px; height: 8px;\n  border-radius: 50%;\n  background: #ef4444;\n  margin-inline-start: 6px;\n}'),
 ("backdrop-filter", "backdrop-filter applies a filter to everything painted behind the element, enabling frosted-glass effects over images or gradients. The element's own background must be semi-transparent for it to be visible.",
  '.modal {\n  background: rgb(15 23 42 / 0.6);\n  backdrop-filter: blur(12px);\n}'),
 ("margin collapsing", "Adjacent vertical margins between block siblings collapse into the larger of the two; a parent with no border/padding also collapses with its first/last child's margin. Flex/grid items never collapse — switching the parent to display:flow-root or flex stops it.",
  '.container { display: flow-root; /* contains child margins */ }'),
 ("transition vs animation", "Transitions animate between two declared states when a property changes (hover, class toggle) and need a trigger; @keyframes animations run independently of state changes, can loop, and control intermediate keyframes. Use transitions for interaction feedback, keyframes for ambient motion.",
  '.btn { transition: transform .2s ease; }\n.btn:hover { transform: translateY(-2px); }'),
 ("logical properties", "margin-inline-start, padding-block-end, inset-inline etc. adapt to the writing direction. In an RTL (dir=\"rtl\") page, margin-inline-start flips automatically — no [dir='rtl'] overrides needed.",
  '.item { margin-inline-start: 1rem; /* right in RTL, left in LTR */ }'),
 ("container queries", "@container lets a component style itself based on its *container's* size instead of the viewport, making truly reusable components. The parent needs container-type: inline-size to establish the query context.",
  '.card-host { container-type: inline-size; }\n@container (min-width: 420px) {\n  .card { display: grid; grid-template-columns: 96px 1fr; }\n}'),
]
JS_FACTS = [
 ("const vs let", "Both are block-scoped and are not redeclared. const additionally forbids reassignment — but objects/arrays held by a const are still mutable inside. Use const by default; switch to let only when reassignment is intended.",
  'const user = { name: "Ava" };\nuser.name = "Rio";   // fine\nuser = {};           // TypeError'),
 ("closures", "A function remembers the scope it was created in, even after that scope's outer function has returned. That is how counters, debouncers and module patterns keep private state.",
  'function counter() {\n  let n = 0;\n  return () => ++n;\n}\nconst next = counter();\nnext(); // 1\nnext(); // 2'),
 ("event delegation", "Instead of attaching a listener to every child, attach one to the parent and inspect event.target.closest(). Fewer listeners, less memory, and it keeps working for elements added later.",
  'list.addEventListener("click", (e) => {\n  const item = e.target.closest("li");\n  if (item) item.classList.toggle("done");\n});'),
 ("async/await", "async functions always return a promise; await pauses the function (not the page) until it settles. Always pair awaits with try/catch or .catch — a rejected promise without handling becomes an unhandled rejection.",
  'async function load() {\n  try {\n    const res = await fetch("/api/items");\n    if (!res.ok) throw new Error(res.status);\n    return await res.json();\n  } catch (err) {\n    console.error("load failed:", err);\n    return [];\n  }\n}'),
 ("event loop", "JavaScript runs one call stack; async callbacks queue as macrotasks, promise continuations as microtasks. The event loop drains ALL microtasks after each macrotask, which is why await continuations run before setTimeout callbacks scheduled at the same time.",
  'console.log("1");\nsetTimeout(() => console.log("2"), 0);   // macrotask\nPromise.resolve().then(() => console.log("3")); // microtask\n// prints: 1, 3, 2'),
 ("debounce", "A debounced function delays execution until input stops for N ms by resetting a timer on every call — ideal for search inputs and resize handlers that must not fire per keystroke.",
  'function debounce(fn, ms = 300) {\n  let t;\n  return (...args) => {\n    clearTimeout(t);\n    t = setTimeout(() => fn(...args), ms);\n  };\n}'),
 ("localStorage", "localStorage stores string key/values synchronously, persists across sessions and is scoped to the origin. Always JSON.stringify on write and parse (with try/catch) on read; never store secrets — any script on the page can read it.",
  'localStorage.setItem("theme", "dark");\nconst theme = JSON.parse(localStorage.getItem("theme") ?? "null");'),
 ("spread vs rest", "Same three dots, two directions: in calls/arrays/objects spread expands elements; in function parameters or destructuring rest collects them into an array/object.",
  'const nums = [3, 1, 2];\nconst sorted = [...nums].sort();          // spread\nfunction sum(...values) { return values.reduce((a, b) => a + b, 0); }'),
 ("optional chaining", "user?.address?.city short-circuits to undefined instead of throwing when an intermediate value is null/undefined. Combine with ?? to provide defaults — but don't hide real bugs by over-using it.",
  'const city = user?.address?.city ?? "Unknown";'),
 ("IntersectionObserver", "IO reports asynchronously when an element intersects the viewport at a given threshold — the efficient backbone of lazy-loading, scroll-reveal and infinite scroll, replacing scroll-event math.",
  'const io = new IntersectionObserver((entries) => {\n  entries.forEach(e => e.isIntersecting && e.target.src === "" && (e.target.src = e.target.dataset.src));\n}, { rootMargin: "200px" });'),
]
HTML_FACTS = [
 ("semantic elements", "header, nav, main, section, article, aside, footer describe the *meaning* of content. They give screen readers landmarks to jump between, improve SEO, and make the DOM self-documenting — divs convey none of this.",
  '<body>\n  <header>…</header>\n  <nav>…</nav>\n  <main>\n    <article>…</article>\n    <aside>…</aside>\n  </main>\n  <footer>…</footer>\n</body>'),
 ("meta viewport", "Without <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"> mobile browsers render at ~980px and zoom out, so media queries never fire. It must be in the head for any responsive design.",
  '<head>\n  <meta charset="UTF-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n</head>'),
 ("forms and labels", "Every input needs a <label for> (or wrapping label). It enlarges the tap target, announces the field to screen readers, and clicking it focuses the input — placeholder is not a label.",
  '<label for="email">Email</label>\n<input id="email" name="email" type="email" required>'),
 ("picture/srcset", "srcset with the w descriptor lets the browser pick the best file for its viewport and DPR; <picture> is for *art direction* — swapping crops per breakpoint.",
  '<img src="hero-800.jpg"\n     srcset="hero-400.jpg 400w, hero-800.jpg 800w, hero-1600.jpg 1600w"\n     sizes="(max-width: 600px) 100vw, 50vw"\n     alt="Team working at a whiteboard">'),
 ("dialog element", "<dialog> ships with showModal()/close(), gives you a top-layer that cannot be overlapped by z-index tricks, traps focus, and pairs with ::backdrop for the overlay.",
  '<dialog id="dlg">\n  <p>Confirm delete?</p>\n  <button onclick="dlg.close()">Cancel</button>\n</dialog>\n<script>document.getElementById("dlg").showModal();</script>'),
]

def explain_fact(facts, fa, cat, cid, diffs):
    topic, expl, demo = random.choice(facts)
    if fa:
        qs = [f"«{topic}» در CSS چطور کار می‌کند؟", f"لطفاً {topic} را با مثال توضیح بده", f"تفاوت {topic} با بقیه در چیست؟ یک مثال بزن"]
        a = f"{expl}\n\n```css\n{demo}\n```" if "css" not in demo[:20] else f"{expl}\n\n{demo}"
        add(cat, "fa", random.choice(diffs), random.choice(qs), a, cid=cid)
    else:
        qs = [f"How does {topic} work?", f"Explain {topic} with a code example",
              f"What is the catch with {topic}? Show me.", f"Quick explainer on {topic}?"]
        lang = "css" if "css" in cat else ("js" if "js" in cat else "html")
        a = f"{expl}\n\n```{lang}\n{demo}\n```"
        add(cat, "en", random.choice(diffs), random.choice(qs), a, cid=cid)

DIF = ["beginner", "intermediate", "advanced"]
for i in range(40):
    explain_fact(CSS_FACTS, False, "en_css", f"tpl-css-en-{i:03d}", DIF)
for i in range(20):
    explain_fact(CSS_FACTS, True, "fa_css", f"tpl-css-fa-{i:03d}", DIF)
for i in range(35):
    explain_fact(JS_FACTS, False, "en_js", f"tpl-js-en-{i:03d}", DIF)
for i in range(15):
    explain_fact(JS_FACTS, True, "fa_js", f"tpl-js-fa-{i:03d}", DIF)
for i in range(20):
    explain_fact(HTML_FACTS, False, "en_html", f"tpl-html-en-{i:03d}", DIF)

# ---------------------------------------------------------------- debugging templates
BUGS = [
 ("flex centering fails", "The items refuse to center.",
  "align/justify are set on the *items* instead of the flex **container**, or the container itself isn't flex.",
  '.wrapper { display: flex; justify-content: center; align-items: center; min-height: 100vh; }',
  '.wrapper { align-items: center; } /* wrapper has no display:flex */',
  "Set flex properties on the container, not the children; verify with DevTools that the computed display is actually flex."),
 ("horizontal scrollbar mobile", "A horizontal scrollbar appears on mobile only.",
  "Some element (often a full-width hero or fixed-width child) exceeds the viewport, or 100vw includes the scrollbar width.",
  'img, video { max-width: 100%; }\nbody { overflow-x: clip; }\n.hero { width: 100%; }',
  '.hero { width: 100vw; } /* 100vw can overflow when a scrollbar exists */',
  "Audit with document.documentElement.scrollWidth vs clientWidth in the console to find the offender."),
 ("z-index not working", "z-index: 9999 does nothing.",
  "The element is not positioned, or a parent creates a new stacking context (transform, opacity, filter, will-change) that caps the child's reach.",
  '.modal { position: fixed; z-index: 100; }\n.parent { transform: none; }',
  '.modal { z-index: 100; } /* no position → ignored */',
  "In DevTools, check 'stacking context' on ancestors; remove the creating property or move the modal to <body>."),
 ("font not loading", "The Google Font never applies.",
  "The stylesheet URL is malformed, preconnect is missing crossorigin, or the family name/weights are wrong; also check the CSS cascade overrides it later.",
  '<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">\n<style>body { font-family: "Inter", system-ui, sans-serif; }</style>',
  '<link href="https://fonts.googleapis.com/css?family=Inter" rel="stylesheet"> /* old API + no display */',
  "Open DevTools → Network → filter 'font' to confirm the woff2 files download, and verify the exact family name."),
 ("null addEventListener", "Uncaught TypeError: Cannot read properties of null (reading 'addEventListener').",
  "The script runs before the DOM element exists (script in <head> without defer) or the selector has a typo.",
  '<script src="app.js" defer></script>',
  '<head><script src="app.js"></script></head> /* runs before <body> exists */',
  "Use defer (or put scripts at the end of body), and null-check the element in development."),
 ("100vh mobile jump", "100vh sections jump when the mobile URL bar hides.",
  "On mobile browsers 100vh is the *largest* viewport (URL bar hidden), so elements resize when it collapses.",
  '.hero { min-height: 100svh; }',
  '.hero { height: 100vh; }',
  "Use svh/lvh/dvh units; 100dvh tracks the dynamic viewport live."),
 ("transition on display", "The transition never animates when toggling visibility.",
  "display: none/grid jumps discretely — it is not animatable, so the browser skips the transition.",
  '.panel { opacity: 0; visibility: hidden; transition: opacity .3s, visibility .3s; }\n.panel.open { opacity: 1; visibility: visible; }',
  '.panel { display: none; transition: opacity .3s; }',
  "Animate opacity/transform/visibility (or use @starting-style / transition-behavior: allow-discrete)."),
 ("event fires twice", "My click handler runs twice.",
  "The listener is attached twice (e.g., re-run init after re-render) or nested handlers both bubble to ancestors.",
  'btn.replaceWith(btn.cloneNode(true)); // drop old listeners, then re-attach\nbtn.addEventListener("click", onClick);',
  'btn.addEventListener("click", onClick); // called on every render',
  "Guard with a bound flag, use { once: true } for one-shot logic, or delegate from a stable parent."),
]
def debug_example(en, cid):
    name, symptom, cause, good, bad, tip = random.choice(BUGS)
    if en:
        u = random.choice([f"{symptom} Why?", f"Help — {symptom.lower()[0]}{symptom[1:].lower()}", f"Debug this: {symptom.lower()}"])
        a = (f"**Symptom:** {symptom}\n\n**Root cause:** {cause}\n\n**Fix:**\n```css\n{good}\n```\n\n**Broken version:**\n```css\n{bad}\n```\n\n**Prevention:** {tip}")
        add("debug", "en", "intermediate", u, a, cid=cid)
    else:
        u = random.choice([f"چرا {symptom}؟", f"این باگ رو درست کن: {symptom}", f"کد من {symptom} — علت چیه؟"])
        a = (f"**علت:** {cause}\n\n**کد درست:**\n```css\n{good}\n```\n\n**کد خراب:**\n```css\n{bad}\n```\n\n**جلوگیری:** {tip}")
        add("fa_debug", "fa", "intermediate", u, a, cid=cid)
for i in range(24): debug_example(True, f"tpl-dbg-en-{i:03d}")
for i in range(18): debug_example(False, f"tpl-dbg-fa-{i:03d}")

# ---------------------------------------------------------------- glossary / terms
TERMS = [("viewport", "نمای دید"), ("breakpoint", "نقطهٔ شکست"), ("dropdown", "منوی بازشو"),
         ("placeholder", "متن راهنما (placeholder)"), ("state", "وضعیت"), ("callback", "تابع برگشتی (callback)"),
         ("responsive", "واکنش‌گرا"), ("tooltip", "راهنمای شناور"), ("sticky header", "هدر چسبان"),
         ("margin/padding", "حاشیهٔ بیرونی/حاشیهٔ درونی"), ("event listener", "شنوندهٔ رویداد"),
         ("scrollbar", "نوار اسکرول"), ("dark mode", "حالت تاریک"), ("lazy loading", "بارگذاری تنبل"),
         ("framework", "چارچوب"), ("deployment", "استقرار"), ("accessibility", "دسترس‌پذیری"),
         ("routing", "مسیریابی"), ("cache", "حافظهٔ نهان"), ("validation", "اعتبارسنجی")]
for i, (en, fa) in enumerate(TERMS):
    lang = random.choice(["fa", "en"])
    if lang == "fa":
        u = random.choice([f"«{en}» در فارسی وب‌دولوپری چه می‌گویند؟", f"معادل فارسی «{en}» چیست؟"])
        a = f"معادل رایج در متن‌های فارسی، «{fa}» است؛ اما در گفتار روزمرهٔ توسعه‌دهنده‌ها همان واژهٔ انگلیسی «{en}» هم بسیار شنیده می‌شود. مثال در جمله: «این {fa} را روی موبایل کوچک‌تر کن.»"
    else:
        u = random.choice([f'How do Persian-speaking developers say "{en}"?',
                           f'Translate "{en}" to Persian for a technical audience.'])
        a = f'The common Persian technical term is «{fa}». In casual dev conversation the English word "{en}" is also used as-is. Example: «این {fa} باید در موبایل هم کار کند.»'
    add("bi_terms", lang, "beginner", u, a, cid=f"tpl-term-{i:03d}")

# ---------------------------------------------------------------- teaching progressions
TEACH = [
 ("flexbox", ['display:flex turns the element into a flex container and children become flex items laid on one axis.',
              'Add justify-content/align-items to align on the main/cross axes, and gap for spacing without margins.',
              'flex: 1 1 0 on items distributes free space; combine with flex-wrap for responsive rows without media queries.',
              'Master flex-basis vs width interplay, order, and using auto margins (margin-inline-start:auto) to push single items.']),
 ("async/await", ['A promise is a placeholder for a future value; .then() chains consume it.',
                  'async/await is syntax sugar: await pauses the async function until the promise settles, making code read top-to-bottom.',
                  'Wrap awaits in try/catch and check response.ok — fetch does NOT throw on HTTP 404/500.',
                  'Run independent awaits concurrently with Promise.all and understand microtask ordering vs setTimeout.']),
 ("CSS grid", ['display:grid + grid-template-columns creates two-dimensional layouts with named tracks.',
               'Use gap, repeat() and fr units; place items with grid-column/row spans.',
               'Define named areas with grid-template-areas for layout-as-diagram readability.',
               'Combine auto-fit/minmax for intrinsic responsiveness, subgrid for nested alignment, and grid for page + flex for components.']),
 ("DOM events", ['addEventListener attaches handlers; the event object carries target and data.',
                 'Understand bubbling/capturing and stopPropagation; use delegation for dynamic children.',
                 'Control behavior with options: { once, passive, capture, signal }.',
                 'Design event-driven modules: custom events with detail payloads, AbortController to detach cleanly.']),
]
for i, (topic, levels) in enumerate(TEACH):
    for rep in range(3):
        lang = "fa" if rep == 2 else "en"
        if lang == "en":
            u = random.choice([f"Teach me {topic} from beginner to advanced.", f"Take me from zero to advanced in {topic}."])
            a = (f"## [Beginner]\n{levels[0]}\n```css\n/* try it in DevTools on any page */\n```\n\n"
                 f"## [Intermediate]\n{levels[1]}\n\n## [Advanced]\n{levels[2]}\n\n## [Expert]\n{levels[3]}\n\n"
                 f"Practice order: rebuild one layout per level; break it on purpose, then fix it.")
        else:
            u = f"«{topic}» را از مبتدی تا پیشرفته به من یاد بده."
            a = (f"## [مبتدی]\n{levels[0]}\n\n## [متوسط]\n{levels[1]}\n\n## [پیشرفته]\n{levels[2]}\n\n## [حرفه‌ای]\n{levels[3]}\n\n"
                 f"تمرین: در هر سطح یک چیدمان واقعی بساز، عمداً خرابش کن و بعد درستش کن.")
        add("pro_teach", lang, "advanced" if rep else "intermediate", u, a, cid=f"tpl-teach-{i}-{rep}")

# ---------------------------------------------------------------- multi-file projects
def multifile(cid, fa=False):
    pal = random.choice(PALETTES)
    brand = random.choice(BRANDS_FA if fa else BRANDS_EN)
    f0, _ = fonts_link(fa)
    tree = ("project/\n├── index.html\n├── css/\n│   └── style.css\n├── js/\n│   └── app.js\n└── README.md")
    html = f'''<!DOCTYPE html>
<html {'lang="fa" dir="rtl"' if fa else 'lang="en"'}>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{brand}</title>
  <link rel="stylesheet" href="css/style.css">
  <script src="js/app.js" defer></script>
</head>
<body>
  <header class="nav"><a class="brand" href="#">{brand}</a><nav><a href="#home">{'خانه' if fa else 'Home'}</a><a href="#about">{'درباره' if fa else 'About'}</a><a href="#contact">{'تماس' if fa else 'Contact'}</a></nav></header>
  <main id="home">
    <section class="hero"><h1>{random.choice(FA_WORDS['hero_t'] if fa else EN_WORDS['hero_t'])}</h1><p>{random.choice(FA_WORDS['hero_s'] if fa else EN_WORDS['hero_s'])}</p></section>
    <section id="about" class="cards"><article class="card"><h3>{'سریع' if fa else 'Fast'}</h3><p>{'عملکرد بالا' if fa else 'Top performance'}</p></article><article class="card"><h3>{'امن' if fa else 'Secure'}</h3><p>{'محافظت کامل' if fa else 'Fully protected'}</p></article></section>
  </main>
  <footer id="contact"><p>© 2025 {brand}</p></footer>
</body>
</html>'''
    css = f''':root {{
  --bg: {pal[1]}; --surface: {pal[4]}; --text: {pal[3]}; --accent: {pal[2]};
}}
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: '{f0}', {("'Vazirmatn', Tahoma, sans-serif" if fa else "system-ui, sans-serif")}; background: var(--bg); color: var(--text); }}
.nav {{ display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; position: sticky; top: 0; background: var(--surface); }}
.nav a {{ color: var(--text); text-decoration: none; margin-inline-start: 1rem; }}
.brand {{ font-weight: 800; font-size: 1.2rem; margin-inline-start: 0 !important; }}
.hero {{ min-height: 60vh; display: grid; place-content: center; text-align: center; gap: 1rem; padding: 2rem; }}
.hero h1 {{ font-size: clamp(1.9rem, 4.5vw + 1rem, 3.6rem); }}
.cards {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); padding: 2rem 1.5rem; max-width: 1000px; margin-inline: auto; }}
.card {{ background: var(--surface); border-radius: 12px; padding: 1.5rem; }}
.card h3 {{ color: var(--accent); }}
footer {{ padding: 2rem; text-align: center; opacity: .7; }}'''
    js = f'''// {brand} — interactions
document.querySelectorAll('a[href^="#"]').forEach((a) => {{
  a.addEventListener('click', (e) => {{
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {{
      e.preventDefault();
      target.scrollIntoView({{ behavior: 'smooth' }});
    }}
  }});
}});'''
    fa_txt = "یک پروژهٔ چندفایلی برای «%s» بساز: index.html، css/style.css و js/app.js با اسکرول نرم." % brand
    en_txt = random.choice([f"Generate a complete multi-file project for {brand}: index.html, css/style.css, js/app.js with smooth scrolling.",
                            f"Scaffold a small static site for {brand} — separate HTML/CSS/JS files, correct relative paths."])
    u = fa_txt if fa else en_txt
    a = (f"**File tree**\n```text\n{tree}\n```\n\n"
         f"**index.html**\n```html\n{html}\n```\n\n"
         f"**css/style.css**\n```css\n{css}\n```\n\n"
         f"**js/app.js**\n```js\n{js}\n```\n\n"
         + ("نکته: مسیرها نسبت به index.html نوشته شده‌اند؛ `defer` اجرای اسکریپت را به بعد از ساخت DOM موکول می‌کند."
            if fa else
            "All paths are relative to index.html; `defer` runs the script after DOM construction."))
    add("pro_multifile", "fa" if fa else "en", "advanced", u, a, cid=cid)
for i in range(10): multifile(f"tpl-mf-fa-{i:02d}", fa=True)
for i in range(20): multifile(f"tpl-mf-en-{i:03d}")

# ---------------------------------------------------------------- short concept drills (short answers, both langs)
DRILLS_EN = [
 ("What does defer do on a script tag?", "It downloads the script in parallel with parsing but executes it **after** the HTML is fully parsed, preserving order — so DOM access is safe. `async` executes as soon as it downloads (order not guaranteed)."),
 ("Why does 100vw cause overflow?", "100vw includes the vertical scrollbar's width on desktop, so an element sized to 100vw is wider than the visible content area — producing a horizontal scrollbar. Prefer 100% or `100dvw`/`100svh` units where appropriate."),
 ("inline vs block vs inline-block?", "inline flows with text and ignores width/height; block starts a new line and fills the width; inline-block flows inline but accepts box properties — handy for buttons and badges."),
 ("What is the CSS cascade order?", "Origin/importance → layers → specificity → source order. Later rules with equal specificity win; !important bypasses specificity (avoid it)."),
 ("Why is my form submitting the page?", "A <button> inside a form defaults to type=\"submit\". For action buttons use type=\"button\", or call e.preventDefault() in the submit handler."),
]
DRILLS_FA = [
 ("خاصیت gap در flex چیست و چه فرقی با margin دارد؟", "`gap` فاصلهٔ بین آیتم‌های flex/grid را می‌سازد؛ برخلاف margin، فاصلهٔ ابتدا و انتها را ایجاد نمی‌کند و نیازی به سلکتورهای :first-child برای حذف margin اضافی نیست."),
 ("چرا تصویر من در کارت کشیده می‌آید؟", "چون با width/height صریح نسبت ابعاد عوض شده است. با `object-fit: cover` تصویر بدون تغییر نسبت، کراپ و پر می‌شود: `img { width: 100%; height: 200px; object-fit: cover; }`"),
 ("تفاوت == و === در جاوااسکریپت؟", "`==` قبل از مقایسه نوع‌ها را تبدیل می‌کند ('1' == 1 درست است)؛ `===` هم نوع را هم مقدار را سخت‌گیرانه چک می‌کند. همیشه === بنویسید."),
 ("چطور یک متغیر CSS را از جاوااسکریپت عوض کنم؟", "`document.documentElement.style.setProperty('--accent', '#22c55e')`؛ همه‌جا که var(--accent) استفاده شده بلافاصله به‌روز می‌شود."),
]
for i, (q, a) in enumerate(DRILLS_EN): add("en_html" if "CSS" not in q else "en_css", "en", "beginner", q, a, cid=f"tpl-drillen-{i}")
for i, (q, a) in enumerate(DRILLS_FA): add("fa_css" if "CSS" not in q else "fa_js", "fa", "beginner", q, a, cid=f"tpl-drillfa-{i}")

# ---------------------------------------------------------------- write
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} examples -> {OUT}")
