document.addEventListener("DOMContentLoaded", () => {
  let slideIndex = 0;
  let slidesInitialized = false;

  async function initialiseSlides() {
    try {
      const slideshowContainer = document.getElementById("slideshow-container");
      if (!slideshowContainer) {
        console.error(
          "Slideshow container element not found! Make sure there's an element with id='slideshow-container'",
        );
        return false;
      }
      console.log("Slideshow container found:", slideshowContainer);

      const imagesResponse = await fetch(`/get_images`);
      if (imagesResponse.ok) {
        const imageData = await imagesResponse.json();
        console.log("Image paths:", imageData);

        const images = imageData.images;
        console.log("Number of images to display:", images.length);

        if (images.length === 0) {
          console.warn("No images found to display");
          return false;
        }

        for (let i = 0; i < images.length; i++) {
          const slide = document.createElement("div");
          slide.className = "mySlides fade";
          const img = document.createElement("img");
          img.src = images[i].path;
          img.onerror = function () {
            console.error(`Failed to load image: ${this.src}`);
          };
          slide.appendChild(img);
          slideshowContainer.appendChild(slide);
          console.log(`Added slide ${i + 1} with image: ${img.src}`);
        }
        return true;
      } else {
        console.error(
          "Failed to fetch image paths:",
          imagesResponse.statusText,
        );
        return false;
      }
    } catch (error) {
      console.error("Error fetching images:", error);
      return false;
    }
  }

  function showSlides() {
    let slides = document.getElementsByClassName("mySlides");

    if (slides.length === 0) {
      console.warn("No slides found to display");
      return;
    }

    for (let i = 0; i < slides.length; i++) {
      slides[i].style.display = "none";
    }
    slideIndex++;
    if (slideIndex > slides.length) {
      slideIndex = 1;
    }
    slides[slideIndex - 1].style.display = "block";
    setTimeout(showSlides, 7000); // Change image every 7 seconds
  }

  async function startSlideshow() {
    if (!slidesInitialized) {
      const success = await initialiseSlides();
      if (success) {
        slidesInitialized = true;
        showSlides();
      } else {
        console.error("Failed to initialize slideshow");
      }
    }
  }

  startSlideshow();
});
