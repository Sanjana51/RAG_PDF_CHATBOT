document.addEventListener("DOMContentLoaded", () => {
    console.log("📄 PDF Chat + Summarizer loaded.");
  
    // Highlight new response/summary
    const responseBoxes = document.querySelectorAll(".response-box");
    responseBoxes.forEach(box => {
      box.style.transition = "background-color 0.3s ease";
      box.style.backgroundColor = "#eaf4ff";
      setTimeout(() => {
        box.style.backgroundColor = "";
      }, 2000);
    });
  
    // File upload alert
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
      fileInput.addEventListener("change", (e) => {
        const fileName = e.target.files[0]?.name;
        if (fileName) {
          alert(`✅ Selected file: ${fileName}`);
        }
      });
    }
  
    // Dark mode toggle
    const toggleBtn = document.getElementById("toggle-dark");
    toggleBtn.addEventListener("click", () => {
      document.body.classList.toggle("dark-mode");
      toggleBtn.textContent = document.body.classList.contains("dark-mode") ? "☀️ Light Mode" : "🌙 Dark Mode";
    });
  });
