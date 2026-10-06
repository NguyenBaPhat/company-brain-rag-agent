import { useRef, useState, type KeyboardEvent } from "react";
import { t } from "../i18n";
import type { Language } from "../types";

interface Props {
  lang: Language;
  busy: boolean;
  disabled: boolean;
  maxChars: number;
  onSend: (text: string) => boolean;
  onStop: () => void;
}

export function Composer({ lang, busy, disabled, maxChars, onSend, onStop }: Props) {
  const d = t(lang);
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const value = text.trim();
    if (!value || busy || disabled || value.length > maxChars) return;
    if (onSend(value)) {
      setText("");
      if (ref.current) ref.current.style.height = "auto";
    }
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends, Shift+Enter inserts a newline; ignore Enter while an IME is composing (Vietnamese Telex/VNI!).
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };
  const over = text.length > maxChars;

  return (
    <div className="composer">
      <textarea
        ref={ref}
        value={text}
        rows={1}
        placeholder={d.placeholder}
        aria-label={d.placeholder}
        onKeyDown={onKey}
        onChange={(e) => {
          setText(e.target.value);
          e.target.style.height = "auto";
          e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
        }}
      />
      {text.length > maxChars * 0.8 && <span className={`counter ${over ? "over" : ""}`}>{text.length}/{maxChars}</span>}
      {busy ? (
        <button type="button" className="btn stop" onClick={onStop}>
          {d.stop}
        </button>
      ) : (
        <button type="button" className="btn send" onClick={submit} disabled={disabled || !text.trim() || over}>
          {d.send}
        </button>
      )}
    </div>
  );
}
