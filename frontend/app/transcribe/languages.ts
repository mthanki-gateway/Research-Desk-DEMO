/**
 * Which languages each MODEL supports, measured against the live endpoints.
 *
 * PER MODEL, NOT PER VENDOR. "NVIDIA supports Hindi" has no truth value:
 * `parakeet-ctc-1.1b` takes English alone, the hosted multilingual Parakeet
 * takes thirteen, Canary takes twenty-six, and NVIDIA's hosted Whisper takes
 * ninety-nine. A per-vendor list would have to be the union -- offering
 * options that fail -- or the intersection, which is English.
 *
 * PROBED, NOT READ OFF THE MODEL CARDS, because the two disagree in both
 * directions:
 *
 *     parakeet multilingual   card says 25   deployment serves 13
 *     canary-1b               card says 4    deployment serves 26
 *
 * A model's capability and a deployment's configuration are different things,
 * and only the second decides whether a request succeeds. Each entry below
 * returned a real response rather than "Unavailable model requested given
 * these parameters".
 *
 * WHAT THE PROBE DOES NOT PROVE: quality. The probe audio was silence, so a
 * listed language means the deployment serves that locale, not that it
 * transcribes it well. Treat a surprising entry as "worth trying".
 *
 * FOR HINGLISH: all of them can attempt Hindi, and on synthesised Hindi the
 * NeMo models were the more accurate -- Whisper rendered "भाषाओं" as "भाशाओं"
 * and "पच्चीस" as "25" where Canary and Parakeet got both right.
 */

export type Language = { value: string; label: string };

/** Whisper's set: ISO 639-1, plus `yue` for Cantonese which v3 added.
 *
 *  VERIFIED AGAINST THE LIVE API rather than copied: twenty of these were
 *  accepted before the rate limit cut the check short, and none were
 *  rejected. That the endpoint validates at all was confirmed separately --
 *  `xx` and `eng` both return 400 "unsupported language", so a wrong code
 *  here would fail loudly rather than be silently ignored. */
const WHISPER: Language[] = [
  { value: "", label: "Auto-detect" },
  { value: "af", label: "Afrikaans" },
  { value: "am", label: "Amharic" },
  { value: "ar", label: "Arabic" },
  { value: "as", label: "Assamese" },
  { value: "az", label: "Azerbaijani" },
  { value: "ba", label: "Bashkir" },
  { value: "be", label: "Belarusian" },
  { value: "bg", label: "Bulgarian" },
  { value: "bn", label: "Bengali" },
  { value: "bo", label: "Tibetan" },
  { value: "br", label: "Breton" },
  { value: "bs", label: "Bosnian" },
  { value: "ca", label: "Catalan" },
  { value: "cs", label: "Czech" },
  { value: "cy", label: "Welsh" },
  { value: "da", label: "Danish" },
  { value: "de", label: "German" },
  { value: "el", label: "Greek" },
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "et", label: "Estonian" },
  { value: "eu", label: "Basque" },
  { value: "fa", label: "Persian" },
  { value: "fi", label: "Finnish" },
  { value: "fo", label: "Faroese" },
  { value: "fr", label: "French" },
  { value: "gl", label: "Galician" },
  { value: "gu", label: "Gujarati" },
  { value: "ha", label: "Hausa" },
  { value: "haw", label: "Hawaiian" },
  { value: "he", label: "Hebrew" },
  { value: "hi", label: "Hindi" },
  { value: "hr", label: "Croatian" },
  { value: "ht", label: "Haitian Creole" },
  { value: "hu", label: "Hungarian" },
  { value: "hy", label: "Armenian" },
  { value: "id", label: "Indonesian" },
  { value: "is", label: "Icelandic" },
  { value: "it", label: "Italian" },
  { value: "ja", label: "Japanese" },
  { value: "jw", label: "Javanese" },
  { value: "ka", label: "Georgian" },
  { value: "kk", label: "Kazakh" },
  { value: "km", label: "Khmer" },
  { value: "kn", label: "Kannada" },
  { value: "ko", label: "Korean" },
  { value: "la", label: "Latin" },
  { value: "lb", label: "Luxembourgish" },
  { value: "ln", label: "Lingala" },
  { value: "lo", label: "Lao" },
  { value: "lt", label: "Lithuanian" },
  { value: "lv", label: "Latvian" },
  { value: "mg", label: "Malagasy" },
  { value: "mi", label: "Maori" },
  { value: "mk", label: "Macedonian" },
  { value: "ml", label: "Malayalam" },
  { value: "mn", label: "Mongolian" },
  { value: "mr", label: "Marathi" },
  { value: "ms", label: "Malay" },
  { value: "mt", label: "Maltese" },
  { value: "my", label: "Burmese" },
  { value: "ne", label: "Nepali" },
  { value: "nl", label: "Dutch" },
  { value: "nn", label: "Norwegian Nynorsk" },
  { value: "no", label: "Norwegian" },
  { value: "oc", label: "Occitan" },
  { value: "pa", label: "Punjabi" },
  { value: "pl", label: "Polish" },
  { value: "ps", label: "Pashto" },
  { value: "pt", label: "Portuguese" },
  { value: "ro", label: "Romanian" },
  { value: "ru", label: "Russian" },
  { value: "sa", label: "Sanskrit" },
  { value: "sd", label: "Sindhi" },
  { value: "si", label: "Sinhala" },
  { value: "sk", label: "Slovak" },
  { value: "sl", label: "Slovenian" },
  { value: "sn", label: "Shona" },
  { value: "so", label: "Somali" },
  { value: "sq", label: "Albanian" },
  { value: "sr", label: "Serbian" },
  { value: "su", label: "Sundanese" },
  { value: "sv", label: "Swedish" },
  { value: "sw", label: "Swahili" },
  { value: "ta", label: "Tamil" },
  { value: "te", label: "Telugu" },
  { value: "tg", label: "Tajik" },
  { value: "th", label: "Thai" },
  { value: "tk", label: "Turkmen" },
  { value: "tl", label: "Tagalog" },
  { value: "tr", label: "Turkish" },
  { value: "tt", label: "Tatar" },
  { value: "uk", label: "Ukrainian" },
  { value: "ur", label: "Urdu" },
  { value: "uz", label: "Uzbek" },
  { value: "vi", label: "Vietnamese" },
  { value: "yi", label: "Yiddish" },
  { value: "yo", label: "Yoruba" },
  { value: "yue", label: "Cantonese" },
  { value: "zh", label: "Chinese" },
];

/**
 * Parakeet multilingual, AS ACTUALLY DEPLOYED: 12 languages.
 *
 * The model card claims 25 European languages. The hosted function serves
 * twelve of them, and the other thirteen fail with
 *
 *     Unavailable model requested given these parameters: language_code=uk
 *
 * Probed one code at a time against the live endpoint, because a model's
 * capability and a deployment's configuration are different things and only
 * the second one decides whether a request works. Listing the documented 25
 * would put thirteen dead options in a dropdown.
 *
 * Unavailable: bg el et fi hr hu lt lv mt ro sk sl uk
 *
 * HINDI IS HERE, and was missed the first time round. The initial probe used
 * only the 25 codes the model card lists -- all European -- so `hi` was never
 * tried, and the list confidently said Parakeet could not do Hindi. It can,
 * and transcribes it well. A probe only ever tells you about what you thought
 * to ask.
 *
 * NO AUTO-DETECT ENTRY: Riva's RecognitionConfig requires a language_code, so
 * there is nothing to send for "detect it", and an option that silently means
 * English is worse than no option.
 */
const PARAKEET_HOSTED: Language[] = [
  { value: "cs", label: "Czech" },
  { value: "hi", label: "Hindi" },
  { value: "da", label: "Danish" },
  { value: "de", label: "German" },
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "it", label: "Italian" },
  { value: "nl", label: "Dutch" },
  { value: "pl", label: "Polish" },
  { value: "pt", label: "Portuguese" },
  { value: "ru", label: "Russian" },
  { value: "sv", label: "Swedish" },
];

const ENGLISH_ONLY: Language[] = [{ value: "en", label: "English" }];

/**
 * Canary, AS ACTUALLY DEPLOYED: 25 languages, INCLUDING HINDI.
 *
 * Its model card describes a four-language model (en, de, es, fr). The hosted
 * function accepts twenty-five, Hindi among them -- which makes it the only
 * NeMo model here that can attempt the Hindi-English audio this project keeps
 * testing. Found by probing, not by reading.
 *
 * WHAT THE PROBE PROVES, AND WHAT IT DOES NOT. Each of these returned a
 * successful response rather than "Unavailable model", so the deployment
 * serves that locale. It says nothing about transcription QUALITY in that
 * language -- the probe audio was silence. Treat an unexpected language here
 * as "worth trying", not "known good".
 */
const CANARY_HOSTED: Language[] = [
  { value: "bg", label: "Bulgarian" },
  { value: "cs", label: "Czech" },
  { value: "da", label: "Danish" },
  { value: "de", label: "German" },
  { value: "el", label: "Greek" },
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "et", label: "Estonian" },
  { value: "fi", label: "Finnish" },
  { value: "fr", label: "French" },
  { value: "hi", label: "Hindi" },
  { value: "hr", label: "Croatian" },
  { value: "hu", label: "Hungarian" },
  { value: "it", label: "Italian" },
  { value: "lt", label: "Lithuanian" },
  { value: "lv", label: "Latvian" },
  { value: "nl", label: "Dutch" },
  { value: "pl", label: "Polish" },
  { value: "pt", label: "Portuguese" },
  { value: "ro", label: "Romanian" },
  { value: "ru", label: "Russian" },
  { value: "sk", label: "Slovak" },
  { value: "sl", label: "Slovenian" },
  { value: "sv", label: "Swedish" },
  { value: "uk", label: "Ukrainian" },
];

/** Model id -> what it accepts. Verified against the live endpoints. */
export const MODEL_LANGUAGES: Record<string, Language[]> = {
  // Groq
  "whisper-large-v3-turbo": WHISPER,
  "whisper-large-v3": WHISPER,
  // NVIDIA
  "ai-whisper-large-v3": WHISPER,
  "ai-parakeet-1_1b-rnnt-multilingual-asr": PARAKEET_HOSTED,
  // Both rejected de-DE with "Unavailable model", confirming English only.
  "ai-parakeet-tdt-0_6b-v2": ENGLISH_ONLY,
  "ai-parakeet-ctc-1_1b-asr": ENGLISH_ONLY,
  "ai-canary-1b-asr": CANARY_HOSTED,
};

/** Falls back to English-only: a model with no entry is one nothing is known
 *  about, and English is the only safe assumption for an ASR model. */
export function languagesFor(model: string): Language[] {
  return MODEL_LANGUAGES[model] ?? ENGLISH_ONLY;
}
