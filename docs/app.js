(() => {
  'use strict';
  const data = window.PROJECT_DATA;
  if (!data) return;
  const $ = (selector, parent = document) => parent.querySelector(selector);
  const create = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const pretty = name => name === 'Shadow' ? 'Input' : name;

  function comparison(input, output, name, eager = false, labels = {}) {
    const beforeLabel = labels.before || 'Input';
    const afterLabel = labels.after || 'Ours';
    const box = create('div', 'compare');
    for (const [source, className, description] of [[output, 'compare-output', labels.afterDescription || 'Shadow removal result'], [input, 'compare-input', labels.beforeDescription || 'Input shadow image']]) {
      const img = create('img', className);
      img.src = source;
      img.alt = `${description}: ${name}`;
      img.loading = eager ? 'eager' : 'lazy';
      img.decoding = 'async';
      img.width = 512;
      img.height = 512;
      box.append(img);
    }
    box.append(create('span', 'compare-label', beforeLabel), create('span', 'compare-label after', afterLabel), create('span', 'compare-handle', '↔'));
    const slider = create('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '100';
    slider.value = '50';
    slider.setAttribute('aria-label', `Before and after comparison for ${name}`);
    slider.setAttribute('aria-valuetext', `50% ${beforeLabel}, 50% ${afterLabel}`);
    slider.addEventListener('input', () => {
      box.style.setProperty('--split', `${slider.value}%`);
      slider.setAttribute('aria-valuetext', `${slider.value}% ${beforeLabel}, ${100 - Number(slider.value)}% ${afterLabel}`);
    });
    box.append(slider);
    return box;
  }

  const dialog = $('#image-dialog');
  let dialogOpener;
  function enlarge(src, title, opener) {
    if (!dialog) return;
    dialogOpener = opener;
    $('img', dialog).src = src;
    $('img', dialog).alt = title;
    $('p', dialog).textContent = title;
    document.body.classList.add('dialog-open');
    dialog.showModal();
  }
  if (dialog) {
    $('.dialog-close', dialog).addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
    dialog.addEventListener('close', () => {
      document.body.classList.remove('dialog-open');
      dialogOpener?.focus();
    });
    document.querySelectorAll('[data-figure]').forEach(button => button.addEventListener('click', () => enlarge(data.figures[button.dataset.figure], $('img', button).alt, button)));
  }

  function imageCell(src, label, scene) {
    const figure = create('figure', `result-cell${label === 'Ours' ? ' ours' : ''}`);
    const button = create('button', 'image-button');
    button.type = 'button';
    button.setAttribute('aria-label', `Enlarge ${label}, ${scene}`);
    const img = create('img');
    img.src = src;
    img.alt = `${label} — ${scene}`;
    img.loading = 'lazy';
    img.decoding = 'async';
    img.width = 512;
    img.height = 512;
    button.append(img);
    button.addEventListener('click', () => enlarge(src, `${label} · ${scene}`, button));
    figure.append(button, create('figcaption', '', label));
    return figure;
  }

  function imageRows(target, items) {
    if (!target) return;
    for (const item of items) {
      const row = create('div', 'synthesis-row');
      row.append(create('p', 'example-label', item.id));
      const grid = create('div', 'image-grid');
      grid.style.setProperty('--cols', Object.keys(item.methods).length);
      for (const [name, src] of Object.entries(item.methods)) grid.append(imageCell(src, pretty(name), item.id));
      row.append(grid);
      target.append(row);
    }
  }

  function synthesisGallery() {
    const target = $('#synthesis-results');
    if (!target) return;
    for (const [index, item] of (window.SYNTHESIS_EXAMPLES || []).slice(0, 8).entries()) {
      const figure = create('figure', 'synthesis-example');
      figure.append(comparison(item.input, item.output, `shadow synthesis example ${index + 1}, ${item.scene}`, false, {
        before: 'Shadow-free',
        after: 'Synthetic shadow',
        beforeDescription: 'Shadow-free source image; white regions are invalid NoData areas',
        afterDescription: 'GSS-Net synthetic shadow image; white regions are invalid NoData areas'
      }));
      const caption = create('figcaption');
      caption.append(create('span', 'synthesis-scene', `${String(index + 1).padStart(2, '0')} · ${item.scene}`));
      figure.append(caption);
      target.append(figure);
    }
  }

  if (document.body.classList.contains('project-page')) {
    synthesisGallery();
    for (const dataset of ['aisd', 'asd']) buildDataset(dataset);

    const tabs = [...document.querySelectorAll('[data-dataset]')];
    function activateTab(tab) {
      tabs.forEach(button => {
        const active = button === tab;
        button.classList.toggle('active', active);
        button.setAttribute('aria-selected', String(active));
        button.tabIndex = active ? 0 : -1;
        $(`#panel-${button.dataset.dataset}`).hidden = !active;
      });
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => activateTab(tab));
      tab.addEventListener('keydown', event => {
        let target;
        if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') target = tabs[(index + 1) % tabs.length];
        if (event.key === 'Home') target = tabs[0];
        if (event.key === 'End') target = tabs[tabs.length - 1];
        if (target) { event.preventDefault(); activateTab(target); target.focus(); }
      });
    });

    $('#copy-citation').addEventListener('click', async () => {
      const text = $('#bibtex').textContent;
      try {
        if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
        else {
          const area = create('textarea');
          area.value = text;
          area.style.cssText = 'position:fixed;left:-9999px;top:0';
          document.body.append(area);
          area.select();
          const copied = document.execCommand('copy');
          area.remove();
          if (!copied) throw new Error('Clipboard unavailable');
        }
        $('#copy-status').textContent = 'BibTeX copied to clipboard.';
        $('#copy-citation').textContent = 'Copied ✓';
        setTimeout(() => { $('#copy-citation').textContent = 'Copy BibTeX'; }, 2400);
      } catch {
        $('#copy-status').textContent = 'Select the BibTeX text above to copy it manually.';
      }
    });
  }

  function buildDataset(dataset) {
    const items = data[dataset];
    const panel = $(`#panel-${dataset}`);
    if (!panel || !items.length) return;
    let scene = 0;
    let detail = false;
    const controls = create('div', 'comparison-controls');
    const sceneLabel = create('label', '', 'Scene');
    const sceneSelect = create('select');
    sceneSelect.id = `${dataset}-scene`;
    sceneLabel.htmlFor = sceneSelect.id;
    items.forEach((item, index) => {
      const option = create('option', '', item.id);
      option.value = String(index);
      sceneSelect.append(option);
    });
    sceneLabel.append(sceneSelect);
    const methodLabel = create('label', '', 'Compare with');
    const methodSelect = create('select');
    methodSelect.id = `${dataset}-method`;
    methodLabel.htmlFor = methodSelect.id;
    methodLabel.append(methodSelect);
    const toggle = create('button', 'detail-toggle', '2× detail');
    toggle.type = 'button';
    toggle.setAttribute('aria-pressed', 'false');
    controls.append(sceneLabel, methodLabel, toggle);
    const grid = create('div', 'image-grid comparison-grid');
    const help = create('p', 'detail-help', 'Click an image to enlarge it, or enable 2× detail to inspect the same region across methods.');
    const pills = create('div', 'scene-pills');
    const pillButtons = items.map((item, index) => {
      const button = create('button', 'scene-pill');
      button.setAttribute('aria-label', `Show ${item.id}`);
      button.addEventListener('click', () => { scene = index; sceneSelect.value = String(index); updateMethods(); });
      pills.append(button);
      return button;
    });
    panel.append(controls, grid, help, pills);
    function render() {
      const item = items[scene];
      grid.replaceChildren(imageCell(item.methods.Shadow, 'Input', item.id), imageCell(item.methods[methodSelect.value], methodSelect.value, item.id), imageCell(item.methods.Ours, 'Ours', item.id));
      pillButtons.forEach((button, index) => {
        button.classList.toggle('active', index === scene);
        button.setAttribute('aria-pressed', String(index === scene));
      });
    }
    function updateMethods() {
      const previous = methodSelect.value;
      methodSelect.replaceChildren();
      Object.keys(items[scene].methods).filter(name => !['Shadow', 'Ours'].includes(name)).forEach(name => {
        const option = create('option', '', name);
        option.value = name;
        methodSelect.append(option);
      });
      methodSelect.value = items[scene].methods[previous] ? previous : items[scene].methods.ShadowFormer ? 'ShadowFormer' : methodSelect.options[0].value;
      render();
    }
    sceneSelect.addEventListener('change', () => { scene = Number(sceneSelect.value); updateMethods(); });
    methodSelect.addEventListener('change', render);
    toggle.addEventListener('click', () => {
      detail = !detail;
      toggle.setAttribute('aria-pressed', String(detail));
      grid.classList.toggle('detail-mode', detail);
      help.textContent = detail ? 'Move over an image to inspect the same region across all three methods.' : 'Click an image to enlarge it, or enable 2× detail to inspect the same region across methods.';
      grid.style.setProperty('--origin', '50% 50%');
    });
    grid.addEventListener('pointermove', event => {
      if (!detail) return;
      const button = event.target.closest('.image-button');
      if (!button) return;
      const rect = button.getBoundingClientRect();
      const x = Math.max(0, Math.min(100, (event.clientX - rect.left) / rect.width * 100));
      const y = Math.max(0, Math.min(100, (event.clientY - rect.top) / rect.height * 100));
      grid.style.setProperty('--origin', `${x}% ${y}%`);
    });
    updateMethods();
  }

  if ($('#scene-list')) buildGallery();
  function buildGallery() {
    const cityFilter = $('#city-filter');
    const search = $('#scene-search');
    const list = $('#scene-list');
    const cities = [...new Set(data.gallery.map(item => item.city))].sort();
    for (const city of cities) {
      const option = create('option', '', city);
      option.value = city;
      cityFilter.append(option);
    }
    let filtered = [...data.gallery];
    const requested = new URLSearchParams(location.search).get('scene');
    let selected = data.gallery.find(item => item.id === requested) || data.gallery.find(item => item.id === 'chicago28_sub10') || data.gallery[0];
    function select(item, updateUrl = true) {
      selected = item;
      $('#scene-title').textContent = item.id;
      $('#scene-meta').textContent = `${item.city} · ${filtered.indexOf(item) + 1} of ${filtered.length} scenes`;
      $('#gallery-comparison').replaceChildren(comparison(item.input, item.output, item.id, true));
      const download = $('#download-result');
      download.href = item.output;
      download.download = `${item.id}-RS-GSSR.webp`;
      list.querySelectorAll('button').forEach(button => {
        const active = button.dataset.scene === item.id;
        button.classList.toggle('active', active);
        if (active) button.setAttribute('aria-current', 'true');
        else button.removeAttribute('aria-current');
      });
      if (updateUrl) {
        try { history.replaceState(null, '', `?scene=${encodeURIComponent(item.id)}`); } catch { /* Direct file viewing still works. */ }
      }
    }
    function filter() {
      const query = search.value.trim().toLowerCase();
      filtered = data.gallery.filter(item => (cityFilter.value === 'all' || item.city === cityFilter.value) && item.id.toLowerCase().includes(query));
      $('#filter-count').textContent = `${filtered.length} of ${data.gallery.length} scenes`;
      list.replaceChildren();
      for (const item of filtered) {
        const button = create('button', 'scene-item');
        button.dataset.scene = item.id;
        button.setAttribute('aria-label', `View ${item.id}`);
        const img = create('img');
        img.src = item.thumb;
        img.alt = '';
        img.loading = 'lazy';
        img.width = 50;
        img.height = 50;
        const text = create('span', '', item.id);
        text.append(create('small', '', item.city));
        button.append(img, text);
        button.addEventListener('click', () => select(item));
        list.append(button);
      }
      const empty = filtered.length === 0;
      $('.gallery-viewer').classList.toggle('empty', empty);
      $('#gallery-empty').hidden = !empty;
      $('#previous-scene').disabled = filtered.length < 2;
      $('#next-scene').disabled = filtered.length < 2;
      if (!empty) select(filtered.includes(selected) ? selected : filtered[0]);
    }
    function navigate(offset) {
      if (!filtered.length) return;
      const index = (filtered.indexOf(selected) + offset + filtered.length) % filtered.length;
      select(filtered[index]);
      const active = list.querySelector('[aria-current=true]');
      if (active && list.scrollHeight > list.clientHeight) list.scrollTop = Math.max(0, active.offsetTop - list.offsetTop - list.clientHeight / 2);
    }
    cityFilter.addEventListener('change', filter);
    search.addEventListener('input', filter);
    $('#previous-scene').addEventListener('click', () => navigate(-1));
    $('#next-scene').addEventListener('click', () => navigate(1));
    filter();
  }
})();
