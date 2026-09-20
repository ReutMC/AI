// ARION ALPHA 1 — Persian booster dataset generator (Persian-phase spec).
// Generates what ParsBench/PersianSyntheticQA lacks:
//   colloquial Persian, multi-turn context retention + requirement changes,
//   explicit language switching (fa/ar/en), fa+en technical mixing,
//   instruction following (rewrite/summarize/explain), troubleshooting dialogs.
// Usage: bun training/booster_gen.mjs [--seconds 900] [--target 2400]
// Appends to datasets/generated/persian_booster.jsonl (resumable, hash-deduped).
import ZAI from 'z-ai-web-dev-sdk';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

const __dirname = path.dirname(new URL(import.meta.url).pathname);
const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'datasets/generated/persian_booster.jsonl');

const args = process.argv.slice(2);
const getArg = (k, d) => { const i = args.indexOf(k); return i >= 0 ? parseInt(args[i + 1], 10) : d; };
const SECONDS = getArg('--seconds', 900);
const TARGET = getArg('--target', 2400);
const CONCURRENCY = 4;
let MIN_INTERVAL_MS = 6000;
const BATCH = 4; // conversations per request
const DEADLINE = Date.now() + SECONDS * 1000;

const SYS = 'تولیدکننده داده آموزشی هستی. فقط JSON خام برگردان، بدون هیچ توضیح اضافه.';

// ---------------- category quotas ----------------
const CATS = {
  colloquial:      { want: 400, desc: 'گفتگوی خودمونی و محاوره‌ای روزمره (سلام، چطوری؟، ممنون). لحن دوستانه و طبیعی، نه رسمی و کتابی.' },
  multi_turn_tech: { want: 500, desc: 'گفتگوی چندنوبته فنی: کاربر درخواست می‌دهد، پاسخ کوتاه سؤال می‌پرسد، کاربر جزئیات می‌دهد/درخواست را عوض می‌کند («نه، منظورم این نبود»، «حالا اینو تغییر بده»)، دستیار متن قبلی را به خاطر دارد.' },
  lang_switch:     { want: 400, desc: 'کنترل زبان: کاربر صریح می‌گوید فارسی/عربی/انگلیسی جواب بده یا وسط حرف زبان را عوض می‌کند. اگر فارسی پرسید فارسی جواب بده، اگر عربی خواست عربی جواب بده، اگر انگلیسی خواست انگلیسی.' },
  fa_en_tech:      { want: 300, desc: 'گفتگوی فنی فارسی+انگلیسی: کاربر فارسی حرف می‌زند ولی اصطلاحات انگلیسی (React، API، deployment، commit) به‌کار می‌برد؛ پاسخ فارسی روان با اصطلاحات فنی انگلیسی در جای درست.' },
  instruct:        { want: 400, desc: 'دستورالعمل: خلاصه‌سازی، بازنویسی (رسمی/خودمونی)، ساده‌توضیح‌دادن، مثال‌زدن، ترجمه فارسی↔انگلیسی، لیست‌کردن مراحل.' },
  troubleshoot:    { want: 400, desc: 'عیب‌یابی: «چرا این کد کار نمی‌کنه؟»، کاربر خطا می‌دهد، دستیار علت را توضیح می‌دهد و راه‌حل قدم‌به‌قدم می‌دهد؛ کد داخل ``` بلوک.' },
};

const TOPICS = [
  'آشپزی و دستور پخت', 'برنامه‌نویسی پایتون', 'JavaScript و React', 'سفر به استانبول',
  'ورزش و باشگاه', 'یادگیری زبان انگلیسی', 'موبایل و گوشی هوشمند', 'کتاب و رمان',
  'موسیقی و ساز', 'باغبانی و گل', 'دانشجویی و امتحان', 'کار و مصاحبه شغلی',
  'سلامت و تغذیه', 'فروشگاه آنلاین', 'هوش مصنوعی', 'وردپرس و سایت‌سازی',
  'دیتابیس SQL', 'لینوکس و ترمینال', 'شبکه و اینترنت', 'گیم و بازی',
  'طراحی و فتوشاپ', 'حسابداری و مالی', 'عکاسی', 'دوچرخه و خودرو',
  'خانه و دکوراسیون', 'بچه و فرزندپروری', 'سینما و سریال', 'اخبار تکنولوژی',
  'استارتاپ و ایده', 'دورکاری و بهره‌وری', 'امتحان آیلتس', 'نویسندگی و وبلاگ',
];

function pickTopics(rng, n) {
  const t = new Set();
  while (t.size < n) t.add(TOPICS[rng.int(TOPICS.length)]);
  return [...t].join('، ');
}

const rng = { int: (n) => Math.floor(Math.random() * n) };

function buildPrompt(cat) {
  const styles = ['خودمونی و صمیمی', 'رسمی و محترمانه', 'ترکیبی از محاوره و فنی'];
  const style = styles[rng.int(3)];
  const turns = cat === 'multi_turn_tech' ? 3 + rng.int(3) : (cat === 'troubleshoot' ? 3 + rng.int(2) : 1 + rng.int(2));
  return `۴ گفتگوی فارسی در دسته «${cat}» بساز.
دسته: ${CATS[cat].desc}
موضوع‌ها (هر گفتگو یکی): ${pickTopics(rng, 4)}
تعداد پیام کاربر در هر گفتگو: ${turns} (یعنی ${turns * 2} پیام)
لحن: ${style}
قوانین مهم:
- فارسی طبیعی و مدرن با نیم‌فاصله درست (می‌شود، نمی‌خواهم) — نه ترجمهٔ تحت‌اللفظی انگلیسی.
- پاسخ‌ها ۲ تا ۸ جمله؛ اگر کد لازم است داخل \`\`\` بگذار.
- در گفتگوی چندنوبته، دستیار باید به جزئیات پیام‌های قبلی اشاره کند و تغییر درخواست کاربر را بپذیرد.
- هیچ placeholder مثل [نام] یا {{}} نگذار. هیچ متن انگلیسی اضافه ننویس.
- اگر دسته lang_switch است حتماً چند نمونه‌ای که کاربر «به عربی جواب بده» یا «به انگلیسی جواب بده» می‌خواهد بگذار و پاسخ هم واقعاً به آن زبان باشد.
فقط یک JSON آرایه برگردان: [{"domain":"موضوع","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}, ...]}, ...]`;
}

function extractConvs(raw) {
  let t = raw.trim();
  const fence = t.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fence) t = fence[1].trim();
  const start = t.indexOf('[');
  const end = t.lastIndexOf(']');
  if (start === -1 || end === -1) throw new Error('no JSON array');
  const arr = JSON.parse(t.slice(start, end + 1));
  if (!Array.isArray(arr)) throw new Error('not array');
  return arr;
}

function validConv(c) {
  if (!c || !Array.isArray(c.messages) || c.messages.length < 2) return false;
  const msgs = c.messages;
  if (msgs[0].role !== 'user' || msgs[msgs.length - 1].role !== 'assistant') return false;
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i];
    if (!m || typeof m.content !== 'string') return false;
    const wantRole = i % 2 === 0 ? 'user' : 'assistant';
    if (m.role !== wantRole) return false;
    if (!m.content.trim()) return false;
    if (m.role === 'assistant' && m.content.trim().length < 40) return false;
    if (m.role === 'user' && m.content.trim().length < 4) return false;
  }
  return true;
}

function normKey(s) {
  return String(s).replace(/[\u064A]/g, 'ی').replace(/[\u0643]/g, 'ک')
    .replace(/\u200c/g, '').replace(/\s+/g, ' ').trim().toLowerCase().slice(0, 180);
}

// ---------------- resume support ----------------
const seen = new Set();
let existing = 0;
if (fs.existsSync(OUT)) {
  for (const line of fs.readFileSync(OUT, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try {
      const j = JSON.parse(line);
      existing++;
      for (const m of j.messages) seen.add(crypto.createHash('md5').update(normKey(m.content)).digest('hex'));
    } catch {}
  }
}
let catCounts = {};
if (fs.existsSync(OUT)) {
  for (const line of fs.readFileSync(OUT, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try { const j = JSON.parse(line); catCounts[j.category] = (catCounts[j.category] || 0) + 1; } catch {}
  }
}

const queue = [];
for (const [cat, meta] of Object.entries(CATS)) {
  const have = catCounts[cat] || 0;
  const need = Math.max(0, Math.min(meta.want, TARGET * meta.want / 2400) - have);
  const batches = Math.ceil(need / BATCH);
  for (let i = 0; i < batches; i++) queue.push({ id: `${cat}_b${i}${existing ? `_r${existing}` : ''}`, cat });
  console.log(`plan ${cat}: have=${have} need≈${need} → ${batches} batches`);
}
console.log(`existing: ${existing}, batches: ${queue.length}`);
if (!queue.length) { console.log('ALL DONE'); process.exit(0); }

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
          { role: 'user', content: buildPrompt(t.cat) },
        ],
        thinking: { type: 'disabled' },
      });
      const raw = c.choices?.[0]?.message?.content || '';
      return extractConvs(raw);
    } catch (e) {
      const m = String((e && e.message) || e || '');
      const is429 = m.includes('429') || m.toLowerCase().includes('too many');
      if (is429) { slowDown(); console.error(`429 ${t.id} a${attempt}: interval=${MIN_INTERVAL_MS}ms`); }
      else console.error(`ERR ${t.id} a${attempt}: ${m.slice(0, 90)}`);
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
      let convs = null;
      try { convs = await genBatch(t); } catch (e) { console.error(`worker${wid}: ${String(e).slice(0, 100)}`); }
      if (!convs) { if (Date.now() > DEADLINE) return; continue; }
      let saved = 0;
      for (const c of convs) {
        try {
          if (!validConv(c)) continue;
          const key = crypto.createHash('md5').update(normKey(c.messages[0].content + '\x00' + c.messages[1].content)).digest('hex');
          if (seen.has(key)) { dup++; continue; }
          seen.add(key);
          const row = {
            id: `${t.id}#${saved}`,
            category: t.cat,
            domain: String(c.domain || 'عمومی').slice(0, 60),
            source: 'booster_llm',
            messages: c.messages.map(m => ({ role: m.role, content: String(m.content).trim() })),
          };
          fs.appendFileSync(OUT, JSON.stringify(row) + '\n');
          saved++; ok++;
          if (ok >= TARGET) break;
        } catch (e) { console.error(`row: ${String(e).slice(0, 100)}`); }
      }
      console.log(`[${new Date().toISOString().slice(11, 19)}] w${wid} ${t.id}: +${saved} (ok=${ok} dup=${dup} fail=${fail} req=${reqCount} left=${queue.length})`);
      if (ok >= TARGET) { console.log('TARGET REACHED'); process.exit(0); }
    }
  } catch (e) { console.error(`worker${wid} fatal: ${String(e).slice(0, 150)}`); }
}

fs.mkdirSync(path.dirname(OUT), { recursive: true });
const q = [...queue];
await Promise.all(Array.from({ length: CONCURRENCY }, (_, i) => worker(q, i)));
console.log(`DONE ok=${ok} dup=${dup} fail=${fail}`);
