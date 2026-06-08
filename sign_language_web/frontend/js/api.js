const API = {
  async health() {
    const res = await fetch('/api/health');
    return await res.json();
  },

  async predictFrame(imageData) {
    const res = await fetch('/api/predict-frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });
    return await res.json();
  },

  async clear() {
    const res = await fetch('/api/clear', { method: 'POST' });
    return await res.json();
  },

  async backspace() {
    const res = await fetch('/api/backspace', { method: 'POST' });
    return await res.json();
  },

  async reset() {
    const res = await fetch('/api/reset', { method: 'POST' });
    return await res.json();
  },

  async predictVideo(file) {
    const formData = new FormData();
    formData.append('video', file);
    formData.append('sample_step', '2');

    const res = await fetch('/api/predict-video', {
      method: 'POST',
      body: formData
    });
    return await res.json();
  }
};
