// ARION ALPHA 1 — v2 dataset generator: complete modern site-building (EN/FA)
// Goal: teach compact, COMPLETE, visually-rich single-file pages + components + debugging.
// Usage: bun scripts/gen_ai_builds.mjs [--seconds 540] [--target 1500]
// Appends to datasets/generated/ai_builds_v2.jsonl (resumable via ids + hash dedupe).
import ZAI from 'z-ai-web-dev-sdk';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

const __dirname = path.dirname(new URL(import.meta.url).pathname);
const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'datasets/generated/ai_builds_v2.jsonl');

const args = process.argv.slice(2);
const getArg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? parseInt(args[i + 1], 10) : d; };
const SECONDS = getArg('--seconds', 540);
const TARGET = getArg('--target', 1500);
const CONCURRENCY = 4;
let MIN_INTERVAL_MS = 8000; // global adaptive throttle between request starts
const BATCH = 1; // single example per request — reliable within output window
const DEADLINE = Date.now() + SECONDS * 1000;

const SYS = 'You are Arion Alpha 1, a professional bilingual (Persian/English) web development AI specialized in HTML, CSS, JavaScript and modern frontend development. Answer in the same language as the user, with complete working code and clear explanations.';

// ---------------- theme matrix for variety ----------------
const THEMES = [
  'dark SaaS with glassmorphism hero', 'minimal light portfolio', 'neo-brutalist agency',
  'elegant restaurant with warm palette', 'tech startup gradient landing', 'premium real-estate listing',
  'medical clinic appointment site', 'fitness club with bold typography', 'online course platform',
  'photography gallery with masonry', 'crypto/fintech dashboard landing', 'eco/green startup',
  'news magazine grid', 'travel agency cards', 'barbershop booking', 'law firm corporate',
  'music festival page', 'coffee shop menu', 'architecture studio', 'kids education platform',
  'pet care service', 'car dealership showcase', 'wedding planner', 'personal blog with sidebar',
];
const FA_THEMES = [
  'فروشگاه اینترنتی با کارت محصول', 'آژانس دیجیتال مارکتینگ', 'کلینیک زیبایی و پزشکی',
  'رزرو رستوران با منوی غذا', 'شرکت معماری و دکوراسیون', 'پلتفرم آموزش آنلاین',
  'سامانه رزرو نوبت پزشک', 'باشگاه ورزشی', 'آژانس مسافرتی', 'فروشگاه لوازم خانگی',
  'استارتاپ فین‌تک', 'مجله خبری تکنولوژی', 'آتلیه عکاسی', 'باشگاه بدنسازی', 'قنادی و کیک',
  'املاک و ساختمان', 'گل فروشی آنلاین', 'تعمیرگاه خودرو', 'کتاب‌فروشی آنلاین', 'استودیو طراحی',
];
const SITES = ['landing page', 'SaaS homepage', 'pricing page', 'portfolio', 'product showcase',
  'contact page with form', 'login/register page', 'blog homepage', 'dashboard overview',
  'checkout summary page', 'FAQ + accordion page', 'team/about page', 'testimonials page',
  'event countdown page', 'features comparison page'];
const COMPONENTS = ['responsive navbar with mobile drawer', 'hero with CTA and stats', 'pricing cards with toggle',
  'testimonials slider (vanilla JS)', 'FAQ accordion', 'tabs component', 'modal with focus trap',
  'data table with sort+search', 'toast notification system', 'image carousel with dots',
  'stats counter on scroll', 'sticky header with scroll progress', 'mega menu', 'form with inline validation',
  'file upload with preview', 'dark/light theme toggle (localStorage)', 'skeleton loaders',
  'infinite scroll list', 'drag-to-reorder list', 'autocomplete search box'];
const APPS = ['todo app with filters + localStorage', 'quiz app with score', 'weather widget (mock data)',
  'notes app with search', 'pomodoro timer', 'calculator with keyboard support', 'expense tracker with chart (CSS bars)',
  'memory card game', 'kanban board with drag&drop', 'currency converter (static rates)'];
const DEBUGS_EN = ['horizontal scroll on mobile', 'z-index not working (stacking context)', 'flexbox centering failing',
  'CSS ignored due to specificity/inheritance', '100vh jump on iOS', 'grid items overflowing', 'transition on display:none',
  'position:absolute escaping parent', 'font not loading (CORS/preload)', 'event listener on dynamically added nodes',
  'fetch returning undefined (async misuse)', 'form submitting despite invalid fields', ' images stretching (object-fit)',
  'margin-collapse surprises', 'blur backdrop not showing'];
const DEBUGS_FA = ['اسکرول افقی در موبایل', 'z-index کار نمی‌کند', 'فلکس وسط‌چین نمی‌شود', 'فونت فارسی لود نمی‌شود',
  'کلاس CSS اعمال نمی‌شود', 'grid بیرون‌زدگی آیتم‌ها', 'مودال بسته نمی‌شود', 'transition روی display کار نمی‌کند',
  'لوکال‌استوریج خطای JSON', 'event روی المنت جدید کار نمی‌کند', 'fetch نتیجه undefined برمی‌گرداند',
  'اعتبارسنجی فرم رد می‌شود', 'عکس‌ها کشیده و بدشکل‌اند', 'backdrop-filter نمایش داده نمی‌شود'];
const TEACH_EN = ['CSS Grid template areas', 'container queries', 'clamp() fluid type', 'custom properties theming',
  'stacking contexts', 'event delegation', 'closures (practical)', 'async/await error handling', 'IntersectionObserver',
  'prefers-reduced-motion', 'semantic HTML & SEO', 'flex vs grid decision', 'position values', 'box-sizing',
  'pseudo-elements ::before/::after', 'CSS specificity full rules', 'fetch + AbortController', 'debounce vs throttle',
  'modules import/export', 'Accessibility: aria & focus'];
const TEACH_FA = ['CSS Grid و template areas', 'container queries', 'تایپوگرافی سیال با clamp()', 'متغیرهای CSS و تم‌سازی',
  'stacking context و z-index', 'event delegation', 'کلوژر در عمل', 'مدیریت خطا با async/await',
  'IntersectionObserver', 'prefers-reduced-motion', 'HTML سمانتیک و سئو', 'انتخاب بین flex و grid',
  'مقادیر position', 'box-sizing', 'شبه‌المان‌ها', 'اولویت CSS به‌زبان ساده', 'fetch و AbortController',
  'debounce در جستجوی زنده', 'ماژول‌ها در جاوااسکریپت', 'دسترس‌پذیری و aria'];

function pick(arr, i) { return arr[i % arr.length]; }

// ---------------- task list (site-build heavy: ~60%) ----------------
const TASKS = [];
let n = 0;
function T(kind, lang, pool, dif) { TASKS.push({ id: `v2-${String(n).padStart(4, '0')}`, kind, lang, pool, dif, n: n++ }); }

// EN complete builds (single-file pages) — 240
for (let i = 0; i < 240; i++) T('build', 'en', { theme: pick(THEMES, i * 7 + 1), site: pick(SITES, i * 3) },
  ['intermediate', 'intermediate', 'advanced'][i % 3]);
// FA complete builds (RTL) — 220
for (let i = 0; i < 220; i++) T('build', 'fa', { theme: pick(FA_THEMES, i * 7 + 3), site: pick(SITES, i * 5 + 2) },
  ['intermediate', 'intermediate', 'advanced'][i % 3]);
// EN components — 120
for (let i = 0; i < 120; i++) T('component', 'en', { c: pick(COMPONENTS, i * 11 + 2), theme: pick(THEMES, i * 3) },
  ['beginner', 'intermediate', 'advanced'][i % 3]);
// FA components — 110
for (let i = 0; i < 110; i++) T('component', 'fa', { c: pick(COMPONENTS, i * 13 + 5), theme: pick(FA_THEMES, i * 3) },
  ['beginner', 'intermediate', 'advanced'][i % 3]);
// EN mini apps — 80
for (let i = 0; i < 80; i++) T('app', 'en', { a: pick(APPS, i * 5 + 1), theme: pick(THEMES, i) },
  ['intermediate', 'advanced'][i % 2]);
// FA mini apps — 70
for (let i = 0; i < 70; i++) T('app', 'fa', { a: pick(APPS, i * 5 + 3), theme: pick(FA_THEMES, i) },
  ['intermediate', 'advanced'][i % 2]);
// EN debugging — 100
for (let i = 0; i < 100; i++) T('debug', 'en', { d: pick(DEBUGS_EN, i * 7 + 2) }, ['intermediate', 'advanced'][i % 2]);
// FA debugging — 100
for (let i = 0; i < 100; i++) T('debug', 'fa', { d: pick(DEBUGS_FA, i * 7 + 2) }, ['intermediate', 'advanced'][i % 2]);
// EN teaching — 90
for (let i = 0; i < 90; i++) T('teach', 'en', { d: pick(TEACH_EN, i * 11 + 4) }, ['beginner', 'intermediate', 'advanced'][i % 3]);
// FA teaching — 90
for (let i = 0; i < 90; i++) T('teach', 'fa', { d: pick(TEACH_FA, i * 11 + 4) }, ['beginner', 'intermediate', 'advanced'][i % 3]);
// mixed-lang (fa user + en terms, short builds) — 70
for (let i = 0; i < 70; i++) T('mixed', 'mixed', { theme: pick(FA_THEMES, i * 5 + 1), site: pick(SITES, i * 7) },
  ['intermediate', 'advanced'][i % 2]);

// ---------------- prompt builder ----------------
const CATS = {
  build: { en: 'en_build', fa: 'fa_build', mixed: 'bi_build' },
  component: { en: 'en_component', fa: 'fa_component', mixed: 'bi_component' },
  app: { en: 'en_app', fa: 'fa_app', mixed: 'bi_app' },
  debug: { en: 'en_debug', fa: 'fa_debug', mixed: 'bi_debug' },
  teach: { en: 'en_teach', fa: 'fa_teach', mixed: 'bi_teach' },
  mixed: { mixed: 'bi_build' },
};

function buildPrompt(t) {
  const fa = t.lang === 'fa' || t.lang === 'mixed';
  const langRule = fa
    ? 'کل پیام کاربر و کل پاسخ دستیار باید به فارسی روان و طبیعی باشد؛ اصطلاحات فنی (flexbox, grid, fetch و…) می‌توانند انگلیسی بمانند. کد HTML باید dir="rtl" و lang="fa" داشته باشد، فونت وزیرمتن با Google Fonts لود شود و متن‌های رابط کاربری فارسی باشند.'
    : 'Both the user message and the assistant answer must be in natural professional English.';

  const CODE_SPEC = fa
    ? `پاسخ دستیار: باید با خط \`\`\`html شروع شود و کد کامل تک‌فایلی (style و script داخل همان فایل) داخل همان بلوک باشد — فشرده ولی کامل: STRICT حداکثر 120 خط کد (~3500 کاراکتر)، بدون TODO و بدون lorem ipsum؛ بعد از بستن \`\`\` حداکثر ۳ بولت کوتاه. کل پاسخ زیر 700 کلمه. کد حتماً داخل بلوک کد مارک‌داون باشد، هرگز HTML خام نده.`
    : `Assistant answer: must START with a line containing \`\`\`html and contain ONE complete single-file HTML document (inline <style> + <script>) inside that fenced block — compact but complete: STRICT max 120 code lines (~3500 chars), real content (no TODO, no lorem ipsum); after closing \`\`\` add max 3 short bullets. Whole answer under 700 words. ALWAYS wrap code in a fenced markdown block, never raw HTML.`;

  let scene;
  switch (t.kind) {
    case 'build':
      scene = fa
        ? `صفحه‌ی «${t.pool.site}» برای کسب‌وکار «${t.pool.theme}» بساز؛ طراحی مدرن، ریسپانسیو کامل، پالت رنگی منسجم، هاور و انیمیشن‌های ظریف.`
        : `Build the ${t.pool.site} for a ${t.pool.theme}; modern design, fully responsive, cohesive palette, subtle hover states and micro-animations.`;
      break;
    case 'component':
      scene = fa
        ? `کامپوننت «${t.pool.c}» را مطابق سبک «${t.pool.theme}» از صفر با HTML/CSS/JS خالص بساز؛ همه‌ی حالت‌ها و تعامل‌ها پیاده شوند.`
        : `Build the "${t.pool.c}" component from scratch (vanilla HTML/CSS/JS) styled for a ${t.pool.theme}; implement ALL states and interactions.`;
      break;
    case 'app':
      scene = fa
        ? `اپلیکیشن کوچک «${t.pool.a}» را در یک فایل HTML بساز؛ داده در localStorage ذخیره شود و UI ریسپانسیو باشد.`
        : `Build a "${t.pool.a}" in ONE html file; persist data in localStorage; responsive UI.`;
      break;
    case 'debug':
      scene = fa
        ? `سناریو: «${t.pool.d}». پیام کاربر مثل یک باگ‌ریپورت واقعی باشد (کد خراب + توضیح علامت). پاسخ: اول ریشه‌ی مشکل، بعد کد اصلاح‌شده‌ی کامل.`
        : `Scenario: "${t.pool.d}". The user message must read like a real bug report (broken snippet + symptom). Answer: root cause first, then the FULL corrected code.`;
      break;
    case 'teach':
      scene = fa
        ? `مفهوم «${t.pool.d}» را آموزش بده: توضیح مفهومی کوتاه، یک مثال کد کوچک ولی قابل‌اجرا، و یک اشتباه رایج.`
        : `Teach "${t.pool.d}": short conceptual explanation, one small but runnable code demo, and one common mistake.`;
      break;
    case 'mixed':
      scene = `The user writes in PERSIAN (casual, English tech terms inline) asking to build the ${t.pool.site} for «${t.pool.theme}». The answer stays in fluent Persian and follows the compact single-file rule: one complete fenced html code block (inline style+script, RTL, Vazirmatn font, max ~200 lines), then max 5 short Persian bullets.`;
      break;
  }

  const userShape = t.kind === 'debug'
    ? (fa ? 'پیام کاربر = باگ‌ریپورت با کد خراب' : 'user message = bug report containing the broken snippet')
    : (fa ? 'پیام کاربر = درخواست طبیعی ۱-۳ جمله‌ای مثل چت واقعی' : 'user message = natural 1-3 sentence request like a real dev chat');

  const items = [0, 1, 2].map(k => ({
    category: CATS[t.kind][t.lang] || 'bi_mixed',
    difficulty: t.dif,
    user: '<user message>',
    assistant: '<assistant answer>',
    hint: k === 0 ? 'short variation' : k === 1 ? 'different wording & details' : 'different again, vary palette/sections',
  }));

  return `${langRule}
Create a JSON ARRAY of exactly ${BATCH} DISTINCT high-quality training examples for a web-development AI.
Theme for all ${BATCH}: ${scene}
${t.kind !== 'mixed' ? CODE_SPEC : ''}
Rules: ${userShape}; vary phrasing/palettes/section names across the ${BATCH} examples; complete working code only; Persian UI text must be real Persian (no transliteration).
Return STRICT JSON only (no markdown fence), exactly an array with ONE item:
[{"category":"${CATS[t.kind][t.lang] || 'bi_mixed'}","difficulty":"${t.dif}","user":"...","assistant":"..."}]`;
}

function salvageObjects(text) {
  // Extract complete top-level {...} objects from a possibly TRUNCATED json array.
  let s = (text || '').replace(/^```[a-z]*\s*/gm, '').replace(/```\s*$/gm, '');
  const start = s.indexOf('[');
  if (start === -1) return [];
  s = s.slice(start + 1);
  const objs = [];
  let depth = 0, inStr = false, esc = false, objStart = -1;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') { inStr = true; continue; }
    if (ch === '{') { if (depth === 0) objStart = i; depth++; }
    else if (ch === '}') { depth--; if (depth === 0 && objStart >= 0) { objs.push(s.slice(objStart, i + 1)); objStart = -1; } }
  }
  const out = [];
  for (const o of objs) { try { out.push(JSON.parse(o)); } catch { /* skip partial */ } }
  return out;
}

function extractExamples(text) {
  let s = (text || '').trim();
  s = s.replace(/^```[a-z]*\s*/gm, '').replace(/```\s*$/gm, '');
  const start = s.indexOf('['), end = s.lastIndexOf(']');
  if (start !== -1 && end > start) {
    try {
      const arr = JSON.parse(s.slice(start, end + 1));
      if (Array.isArray(arr)) return arr.slice(0, BATCH);
    } catch { /* fall through to salvage */ }
  }
  const salv = salvageObjects(s);
  if (salv.length) return salv.slice(0, BATCH);
  throw new Error('no usable json');
}

// ---------------- dedupe ----------------
function norm(s) { return String(s).replace(/\s+/g, ' ').trim().slice(0, 300).toLowerCase(); }
const seen = new Set();
const GEN = path.join(ROOT, 'datasets/generated');
for (const f of fs.readdirSync(GEN).filter(f => f.endsWith('.jsonl'))) {
  for (const line of fs.readFileSync(path.join(GEN, f), 'utf-8').split('\n')) {
    if (!line.trim()) continue;
    try { const j = JSON.parse(line); seen.add(crypto.createHash('md5').update(norm(j.messages?.[1]?.content || '')).digest('hex')); } catch {}
  }
}

// ---------------- resume ----------------
const done = new Set();
if (fs.existsSync(OUT)) {
  for (const line of fs.readFileSync(OUT, 'utf-8').split('\n')) {
    if (!line.trim()) continue;
    try { done.add(JSON.parse(line).id.split('#')[0]); } catch {}
  }
}
const todo = TASKS.filter(t => !done.has(t.id));
console.log(`tasks: ${TASKS.length}, todo: ${todo.length}, dedupe keys: ${seen.size}`);
if (!todo.length) { console.log('ALL DONE'); process.exit(0); }

const zai = await ZAI.create();
let ok = 0, fail = 0, dup = 0, reqCount = 0;
let lastStart = 0;

async function throttle() {
  const now = Date.now();
  const wait = Math.max(0, lastStart + MIN_INTERVAL_MS - now);
  lastStart = now + wait;
  if (wait > 0) await new Promise(r => setTimeout(r, wait));
}

function slowDown() { MIN_INTERVAL_MS = Math.min(25000, Math.round(MIN_INTERVAL_MS * 1.6)); }

async function genBatch(t) {
  let backoff = 25000;
  for (let attempt = 1; attempt <= 5; attempt++) {
    if (Date.now() > DEADLINE) return null;
    try {
      await throttle();
      reqCount++;
      const c = await zai.chat.completions.create({
        messages: [
          { role: 'assistant', content: SYS },
          { role: 'user', content: buildPrompt(t) },
        ],
        thinking: { type: 'disabled' },
      });
      const raw = c.choices?.[0]?.message?.content || '';
      let arr;
      try { arr = extractExamples(raw); }
      catch (pe) {
        console.error(`PARSE ${t.id} a${attempt}: ${String(pe.message).slice(0, 60)} raw=${JSON.stringify(raw.slice(0, 120))}`);
        throw pe;
      }
      return arr;
    } catch (e) {
      const m = String((e && e.message) || e || '');
      const is429 = m.includes('429') || m.toLowerCase().includes('too many');
      if (is429) { slowDown(); console.error(`429 ${t.id} a${attempt}: interval now ${MIN_INTERVAL_MS}ms`); }
      else console.error(`ERR ${t.id} a${attempt}: ${m.slice(0, 80)}`);
      if (attempt === 5) { fail++; console.error(`FAIL ${t.id}`); return null; }
      await new Promise(r => setTimeout(r, is429 ? backoff : 2000 * attempt));
      if (is429) backoff = Math.min(80000, backoff * 1.5);
    }
  }
}

async function worker(queue, wid) {
  try {
    while (queue.length && Date.now() < DEADLINE) {
      const t = queue.shift();
      if (!t) return;
      let arr = null;
      try { arr = await genBatch(t); } catch (e) { console.error(`worker${wid} batch error: ${String(e).slice(0, 100)}`); }
      if (!arr) { if (Date.now() > DEADLINE) return; continue; }
      let saved = 0;
      for (const j of arr) {
        try {
          if (!j || !j.user || !j.assistant) continue;
          if (String(j.user).length < 10 || String(j.assistant).length < 40) continue;
          const key = crypto.createHash('md5').update(norm(j.user)).digest('hex');
          if (seen.has(key)) { dup++; continue; }
          seen.add(key);
          const idx = `${t.id}#${saved}`;
          const row = {
            id: idx,
            category: String(j.category || CATS[t.kind][t.lang] || 'bi_mixed'),
            lang: t.lang === 'mixed' ? 'mixed' : t.lang,
            difficulty: String(j.difficulty || t.dif),
            messages: [
              { role: 'system', content: SYS },
              { role: 'user', content: String(j.user) },
              { role: 'assistant', content: String(j.assistant) },
            ],
          };
          fs.appendFileSync(OUT, JSON.stringify(row) + '\n');
          saved++; ok++;
        } catch (e) { console.error(`row error: ${String(e).slice(0, 100)}`); }
      }
      console.log(`[${new Date().toISOString().slice(11, 19)}] w${wid} ${t.id}: +${saved} (ok=${ok} dup=${dup} fail=${fail} req=${reqCount} left=${queue.length})`);
    }
  } catch (e) { console.error(`worker${wid} fatal: ${String(e).slice(0, 150)}`); }
}

process.on('unhandledRejection', (e) => console.error('unhandled:', String(e).slice(0, 120)));
process.on('uncaughtException', (e) => console.error('uncaught:', String(e).slice(0, 120)));

const queues = Array.from({ length: CONCURRENCY }, () => []);
todo.forEach((t, i) => queues[i % CONCURRENCY].push(t));
await Promise.all(queues.map((q, i) => worker(q, i)));
console.log(`CHUNK DONE ok=${ok} fail=${fail} dup=${dup} req=${reqCount} elapsed=${Math.round((Date.now() - (DEADLINE - SECONDS * 1000)) / 1000)}s`);
process.exit(0);
