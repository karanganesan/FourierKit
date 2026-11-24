/// the prompt the transformers processor builds in `apply_transcription_request`.
enum Prompt {
    /// `<|audio_pad|>`, the positions the engine overwrites with audio embeddings.
    static let audioTokenID: Int32 = 151676

    /// ids for: system turn (empty or the context text), user turn with `audioTokens` audio
    /// placeholders, assistant turn prefilled with "language <Name><asr_text>" when language is given.
    /// an unknown language is left out, as if none were given; resolve it with `languageName(for:)` first
    /// to reject it instead.
    static func transcription(audioTokens: Int, language: String?, context: String?, tokenizer: Tokenizer) -> [Int32] {
        var text = "<|im_start|>system\n" + (context ?? "") + "<|im_end|>\n"
        text += "<|im_start|>user\n<|audio_start|>"
        text += String(repeating: "<|audio_pad|>", count: max(0, audioTokens))
        text += "<|audio_end|><|im_end|>\n"
        // continue_final_message drops the assistant turn's closing "<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        if let language, let name = languageName(for: language) {
            text += "language \(name)<asr_text>"
        }
        return tokenizer.encode(text)
    }

    /// `LANGUAGE_CODE_TO_NAME`, in the processor's order.
    static let languages: [(code: String, name: String)] = [
        ("ar", "Arabic"), ("yue", "Cantonese"), ("zh", "Chinese"), ("cs", "Czech"), ("da", "Danish"),
        ("nl", "Dutch"), ("en", "English"), ("fil", "Filipino"), ("fi", "Finnish"), ("fr", "French"),
        ("de", "German"), ("el", "Greek"), ("hi", "Hindi"), ("hu", "Hungarian"), ("id", "Indonesian"),
        ("it", "Italian"), ("ja", "Japanese"), ("ko", "Korean"), ("mk", "Macedonian"), ("ms", "Malay"),
        ("fa", "Persian"), ("pl", "Polish"), ("pt", "Portuguese"), ("ro", "Romanian"), ("ru", "Russian"),
        ("es", "Spanish"), ("sv", "Swedish"), ("th", "Thai"), ("tr", "Turkish"), ("vi", "Vietnamese"),
    ]

    /// the full name for a code or a name, case-insensitive like the processor's `resolve_language`.
    static func languageName(for code: String) -> String? {
        let key = code.lowercased()
        return languages.first(where: { $0.code.lowercased() == key || $0.name.lowercased() == key })?.name
    }
}
