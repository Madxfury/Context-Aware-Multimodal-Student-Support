(() => {
  const figure = document.querySelector('[data-checkin-sketch]');
  if (!figure) return;

  const buttons = [...figure.querySelectorAll('[data-sketch-stage]')];
  const paths = [...figure.querySelectorAll('[data-sketch-path]')];
  const lengths = paths.map(path => path.getTotalLength());
  const pencil = figure.querySelector('[data-sketch-pencil]');
  const caption = figure.querySelector('[data-sketch-caption]');
  const counter = figure.querySelector('[data-sketch-counter]');
  const still = matchMedia('(prefers-reduced-motion: reduce)');
  const duration = 1480;
  let selected = 0;
  let visible = false;
  let frame = 0;
  let start;
  let drawOnEnter = true;

  function draw(progress) {
    const path = paths[selected];
    const length = lengths[selected];
    path.style.strokeDasharray = String(length);
    path.style.strokeDashoffset = String(length * (1 - progress));
    const tip = path.getPointAtLength(length * progress);
    pencil.setAttribute('transform', 'translate(' + tip.x + ' ' + tip.y + ')');
  }

  function settle() {
    cancelAnimationFrame(frame);
    frame = 0;
    start = undefined;
    draw(1);
  }

  function tick(now) {
    if (start === undefined) start = now;
    const elapsed = Math.min(1, (now - start) / duration);
    draw(elapsed * elapsed * (3 - 2 * elapsed));
    frame = elapsed < 1 ? requestAnimationFrame(tick) : 0;
  }

  function select(stage) {
    selected = stage;
    buttons.forEach((button, index) => {
      button.setAttribute('aria-pressed', String(index === stage));
      paths[index].style.visibility = index === stage ? 'visible' : 'hidden';
    });
    caption.textContent = buttons[stage].dataset.caption;
    counter.textContent = String(stage + 1).padStart(2, '0') + ' / 03';
    settle();
    if (!still.matches && visible) {
      draw(0);
      frame = requestAnimationFrame(tick);
    }
  }

  buttons.forEach((button, index) => {
    button.addEventListener('click', () => select(index));
    button.addEventListener('keydown', event => {
      const directions = { ArrowLeft: -1, ArrowRight: 1, Home: -index, End: buttons.length - 1 - index };
      if (!Object.hasOwn(directions, event.key)) return;
      event.preventDefault();
      const next = (index + directions[event.key] + buttons.length) % buttons.length;
      buttons[next].focus();
      select(next);
    });
  });
  still.addEventListener('change', () => { if (still.matches) settle(); });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      visible = entries.at(-1).isIntersecting;
      if (!visible) settle();
      else if (drawOnEnter) {
        drawOnEnter = false;
        select(selected);
      }
    }, { threshold: 0.35 }).observe(figure);
  } else {
    visible = true;
    select(selected);
  }
})();
