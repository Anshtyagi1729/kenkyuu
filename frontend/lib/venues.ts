/** Venue names as researchers actually say them.
 *
 * Semantic Scholar returns venues in full legal form - "North American Chapter of
 * the Association for Computational Linguistics" is 71 characters, and there are
 * several over 60 in this corpus. Rendered as a badge in a 288px result card, any
 * one of them is wider than the card. Truncation alone would produce "North
 * American Chapter of the…", which is both ugly and ambiguous (it could equally be
 * NAACL or a workshop at it), whereas every researcher in the field reads "NAACL"
 * instantly.
 *
 * Order matters and the list is sorted most-specific-first: NAACL's full name
 * CONTAINS ACL's, and CVPR's contains "Computer Vision", so a shorter pattern placed
 * earlier would capture papers belonging to the longer one. */

type VenueRule = [needle: string, short: string];

const RULES: VenueRule[] = [
  // --- NLP: the NAACL/TACL/ACL ordering is load-bearing ---
  ["north american chapter of the association for computational linguistics", "NAACL"],
  ["transactions of the association for computational linguistics", "TACL"],
  ["annual meeting of the association for computational linguistics", "ACL"],
  ["empirical methods in natural language processing", "EMNLP"],
  ["international conference on computational linguistics", "COLING"],
  ["language resources and evaluation", "LREC"],

  // --- Vision: CVPR before the bare "computer vision" patterns ---
  ["conference on computer vision and pattern recognition", "CVPR"],
  ["computer vision and pattern recognition", "CVPR"],
  ["european conference on computer vision", "ECCV"],
  ["international conference on computer vision", "ICCV"],
  ["applications of computer vision", "WACV"],

  // --- ML / AI ---
  ["neural information processing systems", "NeurIPS"],
  ["international conference on learning representations", "ICLR"],
  ["international conference on machine learning", "ICML"],
  ["artificial intelligence and statistics", "AISTATS"],
  ["aaai conference on artificial intelligence", "AAAI"],
  ["international joint conference on artificial intelligence", "IJCAI"],
  ["knowledge discovery and data mining", "KDD"],
  ["research and development in information retrieval", "SIGIR"],
  ["conference on robot learning", "CoRL"],
  ["international conference on robotics and automation", "ICRA"],
  ["intelligent robots and systems", "IROS"],
  ["conference on fairness, accountability", "FAccT"],
  ["conference on health, inference, and learning", "CHIL"],
  ["conference on human factors in computing systems", "CHI"],

  // --- Journals and transactions ---
  ["transactions on visualization and computer graphics", "TVCG"],
  ["transactions on image processing", "TIP"],
  ["transactions on audio, speech, and language processing", "TASLP"],
  ["transactions on knowledge and data engineering", "TKDE"],
  ["transactions on information systems", "TOIS"],
  ["transactions on pattern analysis and machine intelligence", "TPAMI"],
  ["transactions on neural networks and learning systems", "TNNLS"],
  ["journal of machine learning research", "JMLR"],
  ["trans. mach. learn. res.", "TMLR"],
  ["transactions on machine learning research", "TMLR"],

  // --- Speech / signal ---
  ["acoustics, speech, and signal processing", "ICASSP"],
  ["acoustics, speech and signal processing", "ICASSP"],
  ["international joint conference on neural network", "IJCNN"],
];

/** Venues that are not venues. Every paper in this corpus is on arXiv, so an
 * "arXiv.org" badge distinguishes nothing while occupying the space a real signal
 * would use. Treated as absent rather than shortened. */
const NON_VENUES = new Set(["arxiv.org", "arxiv", "corr", "findings", ""]);

export interface VenueLabel {
  /** What the badge shows - an acronym when recognized, else the original. */
  short: string;
  /** The full name, for the tooltip. */
  full: string;
  /** True when it was matched to a known acronym rather than passed through. */
  abbreviated: boolean;
}

export function abbreviateVenue(venue: string | null): VenueLabel | null {
  if (!venue) return null;
  const trimmed = venue.trim();
  const lower = trimmed.toLowerCase();
  if (NON_VENUES.has(lower)) return null;

  for (const [needle, short] of RULES) {
    if (lower.includes(needle)) return { short, full: trimmed, abbreviated: true };
  }

  // The long tail of smaller venues usually carries its own acronym in trailing
  // parentheses - "2024 4th International Conference on Communication Technology and
  // Information Technology (ICCTIT)" is 96 characters of which the last six are the
  // only part anyone reads. Extracting it handles venues the table will never cover,
  // without inventing an abbreviation for anything.
  //
  // Guarded to look like an acronym (2-10 chars, starts uppercase, no spaces) so a
  // parenthetical that is actually a qualifier - "(Volume 1: Long Papers)" - is left
  // alone rather than becoming the venue's name.
  const parenthetical = trimmed.match(/\(([A-Z][A-Za-z0-9'-]{1,9})\)\s*$/);
  if (parenthetical) {
    return { short: parenthetical[1], full: trimmed, abbreviated: true };
  }

  return { short: trimmed, full: trimmed, abbreviated: false };
}
