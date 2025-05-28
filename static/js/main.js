document.addEventListener("DOMContentLoaded", () => {
  const searchForm = document.getElementById("searchForm");
  const searchInput = document.getElementById("searchInput");
  const searchResultsDiv = document.getElementById("searchResults");

  searchForm?.addEventListener("submit", async (event) => {
    event.preventDefault(); // Prevent the default form submission (page reload)

    const query = searchInput?.value.trim();
    if (!query) {
      if (searchResultsDiv) {
        searchResultsDiv.innerHTML = "<p>Please enter a search query.</p>";
      }
      return;
    }

    if (searchResultsDiv) {
      searchResultsDiv.innerHTML = "<p>Searching...</p>";
    }

    try {
      // Send a GET request to your Flask API endpoint
      const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
      const results = await response.json(); // Parse the JSON response

      displayResults(results);
    } catch (error) {
      console.error("Error fetching search results:", error);
      if (searchResultsDiv) {
        searchResultsDiv.innerHTML =
          "<p>Error fetching search results. Please try again.</p>";
      }
    }
  });

  function displayResults(results) {
    if (!searchResultsDiv) return;

    searchResultsDiv.innerHTML = ""; // Clear previous results

    if (results.length === 0) {
      searchResultsDiv.innerHTML = "<p>No results found.</p>";
      return;
    }

    results.forEach(([term, postings]) => {
      const termSection = document.createElement("div");
      termSection.classList.add("term-section");

      const termHeader = document.createElement("h3");
      termHeader.textContent = `Term: ${term}`;
      termSection.appendChild(termHeader);

      const postingsList = document.createElement("ul");
      postings.forEach(([docId, tf, positions]) => {
        const item = document.createElement("li");
        item.textContent = `Document ID: ${docId}, Frequency: ${tf}, Positions: [${positions.join(", ")}]`;
        postingsList.appendChild(item);
      });

      termSection.appendChild(postingsList);
      searchResultsDiv.appendChild(termSection);
    });
  }
});
