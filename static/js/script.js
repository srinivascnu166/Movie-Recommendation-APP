const searchInput = document.getElementById('search');
const suggestionsDiv = document.getElementById('suggestions');
const recommendationsList = document.getElementById('recommendations');

let timeout = null;

searchInput.addEventListener('input', () => {
  clearTimeout(timeout);
  const query = searchInput.value;

  if (query.length < 2) {
    suggestionsDiv.innerHTML = '';
    return;
  }

  timeout = setTimeout(() => {
    fetch(`/suggest?q=${query}`)
      .then(res => res.json())
      .then(data => {
        suggestionsDiv.innerHTML = '';
        data.forEach(movie => {
          const div = document.createElement('div');
          div.textContent = movie;
          div.onclick = () => {
            searchInput.value = movie;
            suggestionsDiv.innerHTML = '';
            fetch(`/predict?movie=${encodeURIComponent(movie)}`)
              .then(res => res.json())
              .then(recs => {
                console.log('Fetched recommendations:', recs);
                recommendationsList.innerHTML = '';
                (recs.prediction || []).forEach(r => {
                  const li = document.createElement('li');
                  li.textContent = r;
                  recommendationsList.appendChild(li);
                });
              });
          };
          suggestionsDiv.appendChild(div);
        });
      });
  }, 300);
});
