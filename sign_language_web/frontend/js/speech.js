let vietnameseVoice = null;

function loadVietnameseVoice() {
  if (!('speechSynthesis' in window)) return null;

  const voices = window.speechSynthesis.getVoices();

  // Ưu tiên giọng tiếng Việt. Nếu máy không có vi-VN, trình duyệt sẽ fallback sang giọng mặc định.
  vietnameseVoice = voices.find(v => v.lang && v.lang.toLowerCase().startsWith('vi')) || null;
  return vietnameseVoice;
}

function speakVietnamese(text, options = {}) {
  if (!text || text.trim().length === 0) return;

  if (!('speechSynthesis' in window)) {
    alert('Trình duyệt không hỗ trợ phát âm. Hãy dùng Chrome hoặc Edge.');
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'vi-VN';
  utterance.rate = options.rate || 0.95;
  utterance.pitch = options.pitch || 1.0;

  const voice = vietnameseVoice || loadVietnameseVoice();
  if (voice) utterance.voice = voice;

  if (options.cancelBeforeSpeak !== false) {
    window.speechSynthesis.cancel();
  }
  window.speechSynthesis.speak(utterance);
}

if ('speechSynthesis' in window) {
  loadVietnameseVoice();
  window.speechSynthesis.onvoiceschanged = loadVietnameseVoice;
}
