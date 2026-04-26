/**
 * responseNormalizer — Clean text formatting engine
 * 
 * Removes markdown artifacts, meta-commentary, duplicate paragraphs,
 * and conversational filler from LLM responses.
 * 
 * Sentinel-E responses should be: precise, structured, analytical, minimal.
 */

/**
 * Strip markdown syntax from text, preserving structure.
 * Converts markdown to clean plain text suitable for structured rendering.
 */
export function normalizeResponseText(text) {
  if (!text || typeof text !== 'string') return '';

  let cleaned = text;

  // ── Strip internal LLM reasoning/debug tags ──
  // These leak from model outputs and must NEVER be displayed to users.
  const tagPatterns = [
    /<think>[\s\S]*?<\/think>/gi,
    /<thinking>[\s\S]*?<\/thinking>/gi,
    /<analysis>[\s\S]*?<\/analysis>/gi,
    /<reflection>[\s\S]*?<\/reflection>/gi,
    /<internal>[\s\S]*?<\/internal>/gi,
    /<reasoning>[\s\S]*?<\/reasoning>/gi,
    /<system_note>[\s\S]*?<\/system_note>/gi,
    /<meta>[\s\S]*?<\/meta>/gi,
    /\[INTERNAL\][\s\S]*?\[\/INTERNAL\]/gi,
    /\[DEBUG\][\s\S]*?\[\/DEBUG\]/gi,
    /\[SYSTEM\][\s\S]*?\[\/SYSTEM\]/gi,
  ];
  for (const pattern of tagPatterns) {
    cleaned = cleaned.replace(pattern, '');
  }

  // Remove meta commentary / filler phrases
  const fillerPatterns = [
    /^(certainly!?\s*)/i,
    /^(sure!?\s*)/i,
    /^(of course!?\s*)/i,
    /^(great question!?\s*)/i,
    /^(absolutely!?\s*)/i,
    /^(here(?:'s| is) (?:a |an |the |my )?(?:simple |quick |brief )?(?:example|explanation|answer|response|summary|overview|breakdown)(?:\s*(?:of|for|to|:))?[\s:]*)/i,
    /^(let me (?:explain|break down|walk you through|provide|give you)[\s:]*)/i,
    /^(i'd be happy to help[\s!.]*)/i,
    /^(that's a great question[\s!.]*)/i,
  ];

  for (const pattern of fillerPatterns) {
    cleaned = cleaned.replace(pattern, '');
  }

  // NOTE: Markdown syntax (headings, bold, italic, lists, code blocks) is
  // deliberately preserved because the frontend renders via ReactMarkdown.
  // Only strip structural artifacts that break rendering:

  // Remove horizontal rules (triple dashes/asterisks)
  cleaned = cleaned.replace(/^[-*_]{3,}\s*$/gm, '');

  // Remove duplicate blank lines
  cleaned = cleaned.replace(/\n{3,}/g, '\n\n');

  // Remove trailing/leading whitespace
  cleaned = cleaned.trim();

  // Deduplicate paragraphs
  cleaned = deduplicateParagraphs(cleaned);

  return cleaned;
}

/**
 * Remove duplicated paragraphs from text
 */
function deduplicateParagraphs(text) {
  const paragraphs = text.split('\n\n');
  const seen = new Set();
  const unique = [];

  for (const p of paragraphs) {
    const normalized = p.trim().toLowerCase();
    if (normalized.length === 0) continue;
    // Use first 100 chars as dedup key to catch near-duplicates
    const key = normalized.slice(0, 100);
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(p.trim());
    }
  }

  return unique.join('\n\n');
}

/**
 * Detect task complexity to determine whether to show analytics.
 * Simple tasks (code generation, basic questions) should NOT show
 * boundary risk, fragility, evidence pipeline, etc.
 * 
 * Returns: 'simple' | 'moderate' | 'complex'
 */
export function detectTaskComplexity(query) {
  if (!query || typeof query !== 'string') return 'moderate';

  const q = query.toLowerCase().trim();

  // Simple patterns — code gen, basic factual, greetings
  const simplePatterns = [
    /^(write|create|make|generate|code|implement|build)\s+(a\s+)?(simple\s+)?/i,
    /^(hello|hi|hey|good\s+morning|good\s+afternoon)/i,
    /^what\s+(is|are)\s+(a|an|the)\s+\w+\s*\??$/i,
    /^how\s+(?:do\s+(?:i|you)\s+)?(write|make|create|code)/i,
    /^(translate|convert|format|sort|reverse|print|show|list)\s+/i,
    /in\s+(python|javascript|java|c\+\+|go|rust|typescript|ruby|swift)/i,
  ];

  for (const p of simplePatterns) {
    if (p.test(q)) return 'simple';
  }

  // Complex patterns — analysis, strategy, debate, risk assessment
  const complexPatterns = [
    /\b(analyze|evaluate|assess|compare|debate|argue|critique|review)\b/i,
    /\b(strategy|risk|impact|implications|consequences|tradeoffs|trade-offs)\b/i,
    /\b(evidence|prove|verify|validate|fact.?check|source)\b/i,
    /\b(ethical|moral|political|controversial|bias|fairness)\b/i,
    /\b(architecture|system\s+design|scalability|performance)\b/i,
  ];

  let complexScore = 0;
  for (const p of complexPatterns) {
    if (p.test(q)) complexScore++;
  }

  if (complexScore >= 2) return 'complex';
  if (complexScore >= 1) return 'moderate';

  // Length-based heuristic
  if (q.length > 200) return 'moderate';
  if (q.length < 40) return 'simple';

  return 'moderate';
}

/**
 * Determine whether analytics/boundary should be shown
 * based on task complexity and response data
 */
export function shouldShowAnalytics(query, response) {
  const complexity = detectTaskComplexity(query);

  if (complexity === 'simple') return false;

  // Even for moderate tasks, only show if there's meaningful risk
  if (complexity === 'moderate') {
    const boundary = response?.boundary_result || response?.omega_metadata?.boundary_result;
    if (!boundary || (boundary.severity_score != null && boundary.severity_score < 30)) {
      return false;
    }
  }

  return true;
}

const responseNormalizer = { normalizeResponseText, detectTaskComplexity, shouldShowAnalytics };
export default responseNormalizer;
