// ===== Loader =====
document.addEventListener("DOMContentLoaded", () => {
  document.body.classList.add("loading");
  window.addEventListener("load", () => {
    setTimeout(() => {
      document.body.classList.remove("loading");
      document.body.classList.add("loaded");
      document.getElementById("loader")?.classList.add("hidden");
    }, 2500);
    setTimeout(() => {
      document.body.classList.remove("loading");
      document.body.classList.add("loaded");
      document.getElementById("loader")?.classList.add("hidden");
    }, 6000);
  });
});

// ===== Full-page scroll snap JS =====
const sections = document.querySelectorAll("section");
let isScrolling = false;
let currentSection = 0;

// Scroll to a specific section
function scrollToSection(index) {
  if (index < 0) index = 0;
  if (index >= sections.length) index = sections.length - 1;
  isScrolling = true;
  sections[index].scrollIntoView({ behavior: "smooth" });
  currentSection = index;

  // update active links
  document.querySelectorAll(".nav-link").forEach((link, i) => {
    link.classList.toggle("active", i === currentSection);
  });

  // allow next scroll after smooth scroll completes
  setTimeout(() => {
    isScrolling = false;
  }, 700);
}

// Mouse wheel / trackpad scroll
window.addEventListener("wheel", (e) => {
  if (isScrolling) {
    e.preventDefault();
    return;
  }
  if (e.deltaY > 0) {
    scrollToSection(currentSection + 1);
  } else if (e.deltaY < 0) {
    scrollToSection(currentSection - 1);
  }
  e.preventDefault();
}, { passive: false });

// Keyboard arrow keys
window.addEventListener("keydown", (e) => {
  if (isScrolling) return;
  if (e.key === "ArrowDown") scrollToSection(currentSection + 1);
  if (e.key === "ArrowUp") scrollToSection(currentSection - 1);
});

// Nav link clicks
document.querySelectorAll(".nav-link").forEach((link, index) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    scrollToSection(index);
  });
});

// Prevent accidental half scroll on touch devices
let touchStartY = 0;
window.addEventListener("touchstart", (e) => {
  touchStartY = e.touches[0].clientY;
});
window.addEventListener("touchend", (e) => {
  let touchEndY = e.changedTouches[0].clientY;
  if (Math.abs(touchEndY - touchStartY) > 50) {
    if (touchEndY < touchStartY) scrollToSection(currentSection + 1);
    if (touchEndY > touchStartY) scrollToSection(currentSection - 1);
  }
});
const mapBox = document.getElementById("mapBox");

mapBox.addEventListener("click", () => {
  mapBox.classList.toggle("fullscreen");
});
let clockInterval; // To store the interval for the live clock

const weatherButton = document.getElementById("getWeather");

// Function to fetch weather and update UI
function fetchWeather(city, country) {
  const API_KEY = "60113eebfa2a32dd1c9fb1db11859a0c";

  fetch(
    `https://api.openweathermap.org/data/2.5/weather?q=${city},${country}&units=metric&appid=${API_KEY}`
  )
    .then(response => response.json())
    .then(data => {
      if (data.cod !== 200) {
        alert("City not found");
        return;
      }

      // Update weather info
      document.getElementById("w-city").innerText =
        data.name + ", " + data.sys.country;

      document.getElementById("w-temp").innerText =
        Math.round(data.main.temp) + " °C";

      document.getElementById("w-desc").innerText =
        data.weather[0].description;

      const timezoneOffset = data.timezone; // in seconds

      // Clear previous clock interval if exists
      if (clockInterval) clearInterval(clockInterval);

      // Start live clock
      clockInterval = setInterval(() => {
        const utc = Date.now() + new Date().getTimezoneOffset() * 60000; // UTC time in ms
        const localTime = new Date(utc + timezoneOffset * 1000);

        const options = { hour: "2-digit", minute: "2-digit", second: "2-digit" };
        document.getElementById("w-time").innerText =
          "Local Time: " + localTime.toLocaleTimeString("en-US", options);
      }, 1000);
    })
    .catch(() => {
      alert("Error fetching weather data");
    });
}

// Event listener for button click
weatherButton.addEventListener("click", () => {
  const city = document.getElementById("cityInput").value.trim();
  const country = document.getElementById("country").value;

  if (city === "") {
    alert("Please enter a city name");
    return;
  }

  fetchWeather(city, country);
});

// Fetch default weather on page load (Florida, USA)
window.addEventListener("load", () => {
  const defaultCity = "Florida";
  const defaultCountry = "US";
  fetchWeather(defaultCity, defaultCountry);
});
