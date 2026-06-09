const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const backspaceBtn = document.getElementById('backspaceBtn');
const speakBtn = document.getElementById('speakBtn');
const uploadBtn = document.getElementById('uploadBtn');
const videoUpload = document.getElementById('videoUpload');
const videoFileName = document.getElementById("videoFileName");
const expertModeToggle = document.getElementById('expertModeToggle');
const autoSpeakToggle = document.getElementById('autoSpeakToggle');

const serverStatus = document.getElementById('serverStatus');
const cameraState = document.getElementById('cameraState');
const signingState = document.getElementById('signingState');
const currentLabel = document.getElementById('currentLabel');
const confidenceText = document.getElementById('confidenceText');
const confidenceBar = document.getElementById('confidenceBar');
const bufferText = document.getElementById('bufferText');
const top3List = document.getElementById('top3List');
const sentenceText = document.getElementById('sentenceText');
const sentenceChips = document.getElementById('sentenceChips');
const uploadResult = document.getElementById('uploadResult');
const trackingWarning = document.getElementById('trackingWarning');
const autoSpeakState = document.getElementById('autoSpeakState');

let stream = null;
let timer = null;
let isSending = false;
let lastSentence = '';
let autoSpeakEnabled = true;

// REST API demo: gửi frame mỗi 120ms.
// Nếu máy yếu, tăng lên 160-200ms để giảm tải CPU/MediaPipe.
const FRAME_INTERVAL_MS = 120;

function applyMode(isExpert) {
  document.body.classList.toggle('user-mode', !isExpert);
  localStorage.setItem('expertMode', isExpert ? '1' : '0');
}

function initMode() {
  const saved = localStorage.getItem('expertMode');
  const isExpert = saved === null ? true : saved === '1';
  expertModeToggle.checked = isExpert;
  applyMode(isExpert);
}

function applyAutoSpeak(enabled) {
  autoSpeakEnabled = enabled;
  autoSpeakToggle.checked = enabled;
  autoSpeakState.textContent = enabled ? 'Bật' : 'Tắt';
  localStorage.setItem('autoSpeakEnabled', enabled ? '1' : '0');
}

function initAutoSpeak() {
  const saved = localStorage.getItem('autoSpeakEnabled');
  const enabled = saved === null ? true : saved === '1';
  applyAutoSpeak(enabled);
}

async function checkServer() {
  try {
    const data = await API.health();
    serverStatus.textContent = `Server OK | ${data.labels.length} nhãn`;
    serverStatus.className = 'status-pill developer-only ok';
  } catch (e) {
    serverStatus.textContent = 'Server chưa chạy hoặc lỗi model';
    serverStatus.className = 'status-pill developer-only error';
  }
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });
    video.srcObject = stream;
    cameraState.textContent = 'Đang bật';
    renderTrackingWarning(null);
    await API.reset();

    timer = setInterval(captureAndSend, FRAME_INTERVAL_MS);
  } catch (e) {
    alert('Không bật được camera: ' + e.message);
  }
}

function stopCamera() {
  if (timer) clearInterval(timer);
  timer = null;

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.srcObject = null;
  cameraState.textContent = 'Đã dừng';
  renderTrackingWarning(null, 'Camera đã dừng. Bấm Bắt đầu để tiếp tục nhận dạng.');
}

async function captureAndSend() {
  if (!stream || isSending || video.videoWidth === 0) return;

  isSending = true;
  try {
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // JPEG giúp request nhẹ hơn PNG, phù hợp gửi nhiều frame liên tục.
    const imageData = canvas.toDataURL('image/jpeg', 0.72);
    const result = await API.predictFrame(imageData);

    if (result.status === 'error') {
      console.error(result.message);
    } else {
      renderResult(result);
    }
  } catch (e) {
    console.error(e);
  } finally {
    isSending = false;
  }
}

function renderResult(result) {
  signingState.textContent = result.is_signing ? 'Đang ký hiệu' : 'Đang chờ';
  currentLabel.textContent = result.current_display || (result.current_label ? formatLabel(result.current_label) : '---');

  const conf = result.confidence || 0;
  confidenceText.textContent = `${(conf * 100).toFixed(1)}%`;
  confidenceBar.style.width = `${Math.min(conf * 100, 100)}%`;
  bufferText.textContent = `${result.buffer}/${result.sequence_length}`;

  renderTop3(result.top3 || []);
  renderSentence(result.sentence_text, result.sentence_display || []);
  renderTrackingWarning(result.meta || {});

  // Chỉ tự phát âm khi backend xác nhận thêm từ mới vào câu.
  // Không phát âm current_label vì current_label được cập nhật liên tục theo từng frame.
  if (autoSpeakEnabled && result.added_word_display) {
    speakVietnamese(result.added_word_display);
  }
}

function renderTrackingWarning(meta, customMessage = null) {
  if (!trackingWarning) return;

  if (customMessage) {
    trackingWarning.className = 'tracking-warning neutral';
    trackingWarning.textContent = customMessage;
    return;
  }

  if (!meta || Object.keys(meta).length === 0) {
    trackingWarning.className = 'tracking-warning neutral';
    trackingWarning.textContent = 'Bật camera để hệ thống kiểm tra tư thế và bàn tay.';
    return;
  }

  const hasPose = Boolean(meta.has_pose);
  const hasLeft = Boolean(meta.has_left_hand);
  const hasRight = Boolean(meta.has_right_hand);

  if (!hasPose) {
    trackingWarning.className = 'tracking-warning error';
    trackingWarning.textContent = 'Chưa nhận diện được cơ thể. Hãy ngồi/đứng rõ hơn và đưa phần thân trên vào khung hình.';
    return;
  }

  if (!hasLeft && !hasRight) {
    trackingWarning.className = 'tracking-warning warn';
    trackingWarning.textContent = 'Chưa thấy bàn tay. Hãy đưa tay vào giữa khung hình trước khi thực hiện ký hiệu.';
    return;
  }

  if (!hasLeft || !hasRight) {
    trackingWarning.className = 'tracking-warning warn';
    const missing = !hasLeft ? 'tay trái' : 'tay phải';
    trackingWarning.textContent = `Hệ thống chỉ thấy một tay. Nếu ký hiệu cần hai tay, hãy đưa rõ ${missing} vào khung hình.`;
    return;
  }

  trackingWarning.className = 'tracking-warning ok';
  trackingWarning.textContent = 'Tư thế và hai tay đã rõ. Có thể thực hiện ký hiệu.';
}

function renderTop3(items) {
  top3List.innerHTML = '';
  items.forEach(item => {
    const row = document.createElement('div');
    row.className = 'top3-item';
    const label = item.display || formatLabel(item.label);
    row.innerHTML = `<span>${label}</span><strong>${(item.confidence * 100).toFixed(1)}%</strong>`;
    top3List.appendChild(row);
  });
}

function renderSentence(sentence, words = []) {
  lastSentence = sentence || '';
  sentenceText.textContent = lastSentence || 'Chưa có kết quả';

  sentenceChips.innerHTML = '';
  words.forEach(word => {
    const chip = document.createElement('span');
    chip.className = 'word-chip';
    chip.textContent = word;
    sentenceChips.appendChild(chip);
  });
}

function formatLabel(label) {
  return String(label || '').replaceAll('_', ' ');
}

async function clearSentence() {
  const result = await API.clear();
  renderResult(result);
}

async function removeLastWord() {
  const result = await API.backspace();
  renderResult(result);
}

async function uploadVideo() {
  const file = videoUpload.files && videoUpload.files[0];
  if (!file) {
    alert('Hãy chọn video trước.');
    return;
  }

  const oldText = uploadBtn.textContent;
  uploadBtn.disabled = true;
  uploadBtn.textContent = 'Đang xử lý...';
  uploadResult.textContent = '';

  try {
    const result = await API.predictVideo(file);

    if (result.sentence_text) {
      renderSentence(result.sentence_text, result.sentence_display || []);
    } else {
      renderSentence('', []);
    }

    if (expertModeToggle.checked) {
      uploadResult.textContent = JSON.stringify(result, null, 2);
    }
  } catch (e) {
    if (expertModeToggle.checked) {
      uploadResult.textContent = 'Lỗi upload video: ' + e.message;
    } else {
      alert('Không xử lý được video. Hãy thử lại.');
    }
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = oldText;
  }
}



if (videoUpload && videoFileName) {
  videoUpload.addEventListener("change", () => {
    videoFileName.textContent = videoUpload.files[0]?.name || "Chưa chọn video";
  });
}
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
clearBtn.addEventListener('click', clearSentence);
backspaceBtn.addEventListener('click', removeLastWord);
speakBtn.addEventListener('click', () => speakVietnamese(lastSentence));
uploadBtn.addEventListener('click', uploadVideo);
expertModeToggle.addEventListener('change', (e) => applyMode(e.target.checked));
autoSpeakToggle.addEventListener('change', (e) => applyAutoSpeak(e.target.checked));

initMode();
initAutoSpeak();
checkServer();
