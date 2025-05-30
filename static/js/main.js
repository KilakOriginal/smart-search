document.addEventListener("DOMContentLoaded", () => {
  const searchForm = document.getElementById("searchForm");
  const searchInput = document.getElementById("searchInput");
  const searchResultsDiv = document.getElementById("searchResults");

  // Function to perform search and update URL
  async function performSearch(query) {
    if (!query.trim()) {
      if (searchResultsDiv) searchResultsDiv.innerHTML = "<p>Please enter a search term.</p>";
      return;
    }

    // Update URL with the query
    const url = new URL(window.location);
    url.searchParams.set('q', query);
    window.history.pushState({ path: url.href }, '', url.href); // Update URL without full reload

    if (searchResultsDiv) searchResultsDiv.innerHTML = "<p>Searching...</p>";

    try {
      const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const results = await response.json();
      displayResults(results, query);
    } catch (error) {
      console.error("Error fetching search results:", error);
      if (searchResultsDiv) {
        searchResultsDiv.innerHTML =
          "<p>Error fetching search results. Please try again.</p>";
      }
    }
  }

  // Handle form submission
  searchForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = searchInput?.value || "";
    performSearch(query);
  });

  // Populate search bar and perform search if 'q' is in URL on load
  const initialUrlParams = new URLSearchParams(window.location.search);
  const initialQuery = initialUrlParams.get('q');
  if (initialQuery && searchInput) {
    searchInput.value = initialQuery;
    performSearch(initialQuery); // Automatically search if query in URL
  }


  let currentPage = 1;
  const resultsPerPage = 10;
  let allPostings = [];

  function displayResults(results, query) {
    if (!searchResultsDiv) return;

    searchResultsDiv.innerHTML = ""; // Clear previous results
    currentPage = 1; // Reset to first page for new search

    const searchTime = results[0];
    const queryResults = results[1];

    if (queryResults && queryResults.length > 0) {
      allPostings = queryResults;
    } else {
      allPostings = [];
    }

    if (allPostings.length === 0) {
      searchResultsDiv.innerHTML = "<p>No results found.</p>";
      return;
    }

    const searchOverview = document.createElement("div");
    searchOverview.innerHTML = `${allPostings.length} search results found for <b><em>${query}</em></b> in ${searchTime.toFixed(2)} ms`;
    searchOverview.classList.add("search-overview");
    searchResultsDiv.appendChild(searchOverview);

    renderPage();
    renderPaginationControls();
  }

  function renderPage() {
    if (!searchResultsDiv) return;

    const existingResultsContainer = searchResultsDiv.querySelector("#results-container");
    if (existingResultsContainer) {
      existingResultsContainer.innerHTML = "";
    } else {
      const newResultsContainer = document.createElement("div");
      newResultsContainer.id = "results-container";
      searchResultsDiv.appendChild(newResultsContainer);
    }
    
    const resultsContainer = searchResultsDiv.querySelector("#results-container") || searchResultsDiv;

    const startIndex = (currentPage - 1) * resultsPerPage;
    const endIndex = startIndex + resultsPerPage;
    const paginatedPostings = allPostings.slice(startIndex, endIndex);

    const previewPromises = paginatedPostings.map(async ([docId, _]) => {
      const docElement = document.createElement("div");
      docElement.classList.add("document-result");

      const titleElement = document.createElement("h3");
      const titleLink = document.createElement("a");
      titleLink.textContent = `Document ${docId}`;
      titleLink.href = `/document/${docId}`;
      // titleLink.target = "_blank";
      titleElement.appendChild(titleLink);
      docElement.appendChild(titleElement);

      const previewElement = document.createElement("p");
      previewElement.classList.add("document-preview");
      previewElement.textContent = "Loading preview...";
      docElement.appendChild(previewElement);

      try {
        // Fetch preview with default length
        const response = await fetch(`/get_preview/${docId}`);
        if (response.ok) {
          const data = await response.json();
          previewElement.textContent = data.preview || "Preview not available.";
        } else {
          console.error(`Error fetching preview for doc ${docId}: ${response.status}`);
          previewElement.textContent = "Preview could not be loaded.";
        }
      } catch (error) {
        console.error(`Network error fetching preview for doc ${docId}:`, error);
        previewElement.textContent = "Preview load failed (network error).";
      }
      return docElement;
    });

    // Once all promises are resolved or rejected
    Promise.all(previewPromises).then(docElements => {
      // resultsContainer.innerHTML = "";
      docElements.forEach(docEl => {
        resultsContainer.appendChild(docEl);
      });
    }).catch(error => {
        console.error("Error rendering page with previews:", error);
        resultsContainer.innerHTML = "<p>Error displaying page results.</p>";
    });
  }

  function renderPaginationControls() {
    if (!searchResultsDiv) return;

    let paginationDiv = searchResultsDiv.querySelector(".pagination-controls");
    if (paginationDiv) {
      paginationDiv.innerHTML = ""; // Clear existing controls
    } else {
      paginationDiv = document.createElement("div");
      paginationDiv.classList.add("pagination-controls");
      searchResultsDiv.appendChild(paginationDiv);
    }

    const totalPages = Math.ceil(allPostings.length / resultsPerPage);

    if (totalPages <= 1) {
      return;
    }

    // Previous Button
    const previousButton = document.createElement("button");
    previousButton.textContent = "Previous";
    previousButton.classList.add("search-button-style");
    if (currentPage === 1) {
      previousButton.style.display = "none"; // Hide if on the first page
    }
    previousButton.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage--;
        renderPage();
        renderPaginationControls();
      }
    });
    paginationDiv.appendChild(previousButton);

    // Page Number Buttons
    const maxButtonsToDisplay = 10;
    let startPage, endPage;

    if (totalPages <= maxButtonsToDisplay) {
      // Less than or equal to 10 pages, show all
      startPage = 1;
      endPage = totalPages;
    } else {
      // More than 10 pages, calculate window
      const pagesBeforeCurrent = 5; // Aim to have 5 pages before current (making current the 6th)
      const pagesAfterCurrent = maxButtonsToDisplay - 1 - pagesBeforeCurrent; // 4 pages after

      startPage = currentPage - pagesBeforeCurrent;
      endPage = currentPage + pagesAfterCurrent;

      if (startPage < 1) {
        startPage = 1;
        endPage = maxButtonsToDisplay;
      } else if (endPage > totalPages) {
        endPage = totalPages;
        startPage = totalPages - maxButtonsToDisplay + 1;
      }
    }

    for (let i = startPage; i <= endPage; i++) {
      const pageButton = document.createElement("button");
      pageButton.textContent = i;
      pageButton.classList.add("pagination-button");
      if (i === currentPage) {
        pageButton.classList.add("active"); // Highlight current page
      }
      pageButton.addEventListener("click", () => {
        currentPage = i;
        renderPage();
        renderPaginationControls();
      });
      paginationDiv.appendChild(pageButton);
    }

    // Next Button
    const nextButton = document.createElement("button");
    nextButton.textContent = "Next";
    nextButton.classList.add("search-button-style");
    if (currentPage === totalPages) {
      nextButton.style.display = "none"; // Hide if on the last page
    }
    nextButton.addEventListener("click", () => {
      if (currentPage < totalPages) {
        currentPage++;
        renderPage();
        renderPaginationControls();
      }
    });
    paginationDiv.appendChild(nextButton);
  }
});
