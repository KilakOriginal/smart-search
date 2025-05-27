document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm') as HTMLFormElement | null;
    const searchInput = document.getElementById('searchInput') as HTMLInputElement | null;
    const searchResultsDiv = document.getElementById('searchResults') as HTMLDivElement | null;

    searchForm?.addEventListener('submit', async (event: SubmitEvent) => {
        event.preventDefault(); // Prevent the default form submission (page reload)

        const query = searchInput?.value.trim();
        if (!query) {
            if (searchResultsDiv) {
                searchResultsDiv.innerHTML = '<p>Please enter a search query.</p>';
            }
            return;
        }

        if (searchResultsDiv) {
            searchResultsDiv.innerHTML = '<p>Searching...</p>'; // Show a loading message
        }

        try {
            // Send a GET request to your Flask API endpoint
            const response = await fetch(`/search?query=${encodeURIComponent(query)}`);
            const data = await response.json(); // Parse the JSON response

            // Assuming 'results' is an array of objects with 'url' and 'title' properties
            displayResults(data.results as { url: string; title: string }[]);
        } catch (error) {
            console.error('Error fetching search results:', error);
            if (searchResultsDiv) {
                searchResultsDiv.innerHTML = '<p>Error fetching search results. Please try again.</p>';
            }
        }
    });

    function displayResults(results: { url: string; title: string }[]): void {
        if (!searchResultsDiv) return;

        searchResultsDiv.innerHTML = ''; // Clear previous results

        if (results.length === 0) {
            searchResultsDiv.innerHTML = '<p>No results found.</p>';
            return;
        }

        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.classList.add('result-item');
            resultItem.innerHTML = `
                <h3><a href="${result.url}" target="_blank">${result.title}</a></h3>
                <p>${result.url}</p>
            `;
            searchResultsDiv.appendChild(resultItem);
        });
    }
});