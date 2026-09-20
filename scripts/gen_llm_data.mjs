// ARION ALPHA 1 — LLM-powered synthetic data generation (backend only)
// Generates diverse Persian/English webdev instruction examples via GLM.
import ZAI from 'z-ai-web-dev-sdk';
import fs from 'fs';

const OUT = '/home/z/my-project/arion-alpha-1/datasets/generated/llm_data.jsonl';
const TARGET = parseInt(process.env.TARGET || '420', 10);
const CONCURRENCY = 1;

const SYS = 'You are Arion Alpha 1, a professional bilingual (Persian/English) web development AI specialized in HTML, CSS, JavaScript and modern frontend development. Answer in the same language as the user, with complete working code and clear explanations.';

const TASKS = [
  // ---- Persian (fa)
  ...Array.from({ length: 110 }, (_, i) => ({
    id: `llm-fa-${String(i).padStart(3, '0')}`, lang: 'fa',
    topics: ['سمانتیک HTML و سئو','فلکس‌باکس در عمل','CSS Grid','media query و ریسپانسیو','فونت وزیرمتن و تایپوگرافی فارسی','transition و animation','متغیرهای CSS','کار با DOM','event delegation','فرم و اعتبارسنجی','fetch و async/await','localStorage','خطاهای رایج جاوااسکریپت','ساخت کارت محصول','ساخت نوبار ریسپانسیو','ساخت داشبورد ادمین','ساخت فرم لاگین','ساخت صفحهٔ قیمت‌گذاری','دیباگ چیدمان RTL','accessibility در فارسی','بهینه‌سازی تصاویر','scroll و انیمیشن','dark mode','دیاگرام file tree پروژه','بازسازی کد شلخته'],
    dif: ['beginner','intermediate','intermediate','advanced'][i % 4],
  })),
  // ---- English (en)
  ...Array.from({ length: 130 }, (_, i) => ({
    id: `llm-en-${String(i).padStart(3, '0')}`, lang: 'en',
    topics: ['semantic HTML','flexbox patterns','CSS Grid layouts','container queries','custom properties theming','clamp() fluid typography','specificity wars','position & stacking contexts','transitions vs keyframes','pseudo-elements tricks','forms & validation','IntersectionObserver','fetch + error handling','closures in practice','event loop deep dive','debounce & throttle','module patterns','landing page hero','pricing table','testimonial slider','responsive navbar','footer design','login page','register page','dashboard cards','table design','toasts & notifications','modal accessibility','lazy loading images','perf: reflow & repaint','code review a div soup','refactor inline styles','why my CSS is ignored','grid vs flexbox choice','Google Fonts loading','prefers-reduced-motion','file upload styling','character counter','infinite scroll','drag to scroll'],
    dif: ['beginner','intermediate','intermediate','advanced','expert'][i % 5],
  })),
  // ---- Mixed (fa + English terms)
  ...Array.from({ length: 90 }, (_, i) => ({
    id: `llm-mix-${String(i).padStart(3, '0')}`, lang: 'mixed',
    topics: ['چرا این media query کار نمی‌کند','flexbox وسط‌چین نمی‌شود','ارور Cannot read properties of null','فونت فارسی لود نمی‌شود','اسکرول افقی موبایل','z-index کار نمی‌کند','transition با display','grid در RTL','fetch ارور 404','localStorage و JSON','debounce سرچ','responsive تصاویر','backdrop-filter','event delegation','فرم با FormData','حالت تاریک با data-theme','آپلود عکس با preview','تایمر شمارش معکوس','تب‌ها بدون کتابخانه','اسکرول اسموس','gap در فلکس','100vh در آیفون','سئوی فارسی','lazy loading عکس‌ها','tooltips فقط با CSS'],
    dif: ['intermediate','intermediate','advanced'][i % 3],
  })),
  // ---- Explicit switch requests
  ...Array.from({ length: 40 }, (_, i) => ({
    id: `llm-sw-${String(i).padStart(3, '0')}`, lang: i % 2 ? 'fa2en' : 'en2fa',
    topics: ['explain what CSS specificity is','توضیح فلکس‌باکس','explain the event loop','توضیح async/await','explain grid vs flexbox','توضیح media query','explain custom properties','توضیح event delegation','explain box-sizing','توضیح سمانتیک HTML','explain font-display','توضیح transition','explain IntersectionObserver','توضیح fetch','explain closures','توضیح localStorage','explain debounce','توضیح z-index','explain object-fit','توضیح gap'],
    dif: ['intermediate','advanced'][i % 2],
  })),
];

function buildPrompt(t) {
  const langRule = {
    fa: 'Write BOTH the user message and the assistant answer ENTIRELY in natural fluent Persian (English technical terms may stay in English). User writes like a real Iranian developer.',
    en: 'Write BOTH the user message and the assistant answer in natural professional English.',
    mixed: 'Write the user message in PERSIAN containing English technical terms inline (like a real Iranian developer chat), and the answer in fluent Persian keeping technical terms in English.',
    fa2en: 'The user message must be in ENGLISH but ends with a request like "please answer in Persian". The ENTIRE assistant answer must be in PERSIAN.',
    en2fa: 'The user message must be in PERSIAN but ends with a request like «لطفاً به انگلیسی جواب بده». The ENTIRE assistant answer must be in ENGLISH.',
  }[t.lang];

  const kind = ['teaching question with a small code example',
    'a "build me X" request answered with a COMPLETE working single-file code (html/css/js fences)',
    'a debugging question (symptom) answered with root cause + corrected code',
    'a refactor request: flawed snippet in the user message, improved code + explanation in the answer',
    'a conceptual question answered concisely with a tiny demo'][t.id.length % 5];

  return `${langRule}
Create ONE high-quality training example for a web-development AI about: ${t.topics[t.id.length % t.topics.length]} (topic index ${t.id}).
Interaction type: ${kind}.
Difficulty: ${t.difficulty}.
Rules for the assistant answer: complete working code in fenced blocks with language tags; NO TODOs, NO placeholders, NO "implement yourself"; explanation after the code (or diagnosis-first for debugging); explain WHY, not just what; 80-450 words total.
The user message must sound like a real developer (varied phrasing, 1-3 sentences).
Return STRICT JSON only, no markdown fence, exactly:
{"category":"<one of fa_html,fa_css,fa_js,fa_build,fa_debug,en_html,en_css,en_js,en_build,debug,pro_csstricks,pro_jspatterns,pro_teach,bi_mixed,bi_switch>","difficulty":"${t.difficulty}","user":"<user message>","assistant":"<assistant answer>"}`;
}

function extractJSON(text) {
  let s = (text || '').trim();
  if (!s) throw new Error('empty response');
  // strip ALL code fences (handles inner fences in assistant content)
  s = s.replace(/^```[a-z]*\s*/gm, '').replace(/```\s*$/gm, '');
  const start = s.indexOf('{');
  const end = s.lastIndexOf('}');
  if (start === -1 || end === -1) throw new Error('no json in: ' + s.slice(0, 80));
  return JSON.parse(s.slice(start, end + 1));
}

const done = new Set();
if (fs.existsSync(OUT)) {
  for (const line of fs.readFileSync(OUT, 'utf-8').split('\n')) {
    if (!line.trim()) continue;
    try { done.add(JSON.parse(line).id); } catch { /* ignore */ }
  }
}
const todo = TASKS.filter(t => !done.has(t.id)).slice(0, TARGET);
console.log(`todo: ${todo.length} (done before: ${done.size})`);

let ok = 0, fail = 0;
const zai = await ZAI.create();

async function genOnce(t) {
  for (let attempt = 1; attempt <= 30; attempt++) {
    try {
      const c = await zai.chat.completions.create({
        messages: [
          { role: 'assistant', content: SYS },
          { role: 'user', content: buildPrompt(t) },
        ],
        thinking: { type: 'disabled' },
      });
      const raw = c.choices?.[0]?.message?.content || '';
      const j = extractJSON(raw);
      if (!j.user || !j.assistant) throw new Error('missing fields');
      if (String(j.user).length < 10 || String(j.assistant).length < 40) throw new Error('too short');
      return j;
    } catch (e) {
      const is429 = String(e.message).includes('429') || String(e.message).includes('Too many');
      if (attempt === 30) { fail++; console.error(`FAIL ${t.id}: ${e.message}`); return null; }
      await new Promise(r => setTimeout(r, is429 ? 25000 : 1500 * attempt));
    }
  }
}

async function worker(queue) {
  while (queue.length) {
    const t = queue.shift();
    if (!t) return;
    const j = await genOnce(t);
    if (!j) continue;
    const row = {
      id: t.id,
      category: String(j.category || 'bi_mixed'),
      lang: t.lang.startsWith('fa') ? 'fa' : (t.lang === 'en' ? 'en' : 'mixed'),
      difficulty: t.dif,
      messages: [
        { role: 'system', content: SYS },
        { role: 'user', content: String(j.user) },
        { role: 'assistant', content: String(j.assistant) },
      ],
    };
    fs.appendFileSync(OUT, JSON.stringify(row) + '\n');
    ok++;
    if (ok % 10 === 0) console.log(`progress: ${ok} ok, ${fail} fail`);
  }
}

process.on('unhandledRejection', (e) => console.error('unhandled:', String(e).slice(0, 120)));
process.on('uncaughtException', (e) => console.error('uncaught:', String(e).slice(0, 120)));

await Promise.all(Array.from({ length: CONCURRENCY }, () => worker([...todo])));
console.log(`DONE ok=${ok} fail=${fail} total_in_file=${done.size + ok}`);
