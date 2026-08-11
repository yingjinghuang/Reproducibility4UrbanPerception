const header = document.querySelector("[data-header]");

const updateHeader = () => {
  header?.classList.toggle("is-scrolled", window.scrollY > 18);
};

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const revealItems = document.querySelectorAll(".reveal");

if (reduceMotion || !("IntersectionObserver" in window)) {
  revealItems.forEach((item) => item.classList.add("is-visible"));
} else {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8%", threshold: 0.1 },
  );

  revealItems.forEach((item) => observer.observe(item));
}

const copyButton = document.querySelector("[data-copy-citation]");
const citation = document.querySelector("#citation-text");

copyButton?.addEventListener("click", async () => {
  const originalLabel = copyButton.textContent;
  try {
    await navigator.clipboard.writeText(citation?.textContent.trim() ?? "");
    copyButton.textContent = "Copied";
  } catch {
    copyButton.textContent = "Select to copy";
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }

  window.setTimeout(() => {
    copyButton.textContent = originalLabel;
  }, 1800);
});
