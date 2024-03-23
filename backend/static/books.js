let books = [];

fetch('/suggest')
  .then((response) => response.json())
  .then((data) => (albums = data));