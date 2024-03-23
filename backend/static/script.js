// getting all required elements
const searchWrapper = document.querySelector('input-box');
const inputBox = searchWrapper.querySelector('input');
const suggBox = searchWrapper.querySelector('.autocom-box');
const icon = searchWrapper.querySelector('.icon');
let linkTag = searchWrapper.querySelector('a');
let webLink;

document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    searchWrapper.classList.remove('active');
    query();
    document.activeElement.blur();
    return;
  }
});

// search icon onclick
icon.onclick = () => {
  searchWrapper.classList.remove('active');
  query();
  return;
};

// if user press any key and release
inputBox.onkeyup = (e) => {
  // if (e.keyCode === 13) {
  //   searchWrapper.classList.remove('active');
  //   query();
  //   return;
  // }

  let userData = e.target.value; //user enetered data

  let emptyArray = [];
  if (userData) {
    emptyArray = albums.filter((data) => {
      //filtering array value and user characters to lowercase and return only those words which are start with user enetered chars
      return data.toLocaleLowerCase().includes(userData.toLocaleLowerCase());
    });
    emptyArray = emptyArray.map((data) => {
      // passing return data inside li tag
      return (data = `<li>${data}</li>`);
    });
    searchWrapper.classList.add('active'); //show autocomplete box
    showSuggestions(emptyArray);
    let allList = suggBox.querySelectorAll('li');
    for (let i = 0; i < allList.length; i++) {
      //adding onclick attribute in all li tag
      allList[i].setAttribute('onclick', 'select(this)');
    }
  } else {
    searchWrapper.classList.remove('active'); //hide autocomplete box
  }
};

function select(element) {
  let selectData = element.textContent;
  inputBox.value = selectData;
  searchWrapper.classList.remove('active');
  inputBox.focus();
}

function showSuggestions(list) {
  let listData;
  if (!list.length) {
    userValue = inputBox.value;
    listData = `<li>${userValue}</li>`;
  } else {
    listData = list.join('');
  }
  suggBox.innerHTML = listData;
}

// if user unfocuses from search box
inputBox.addEventListener('focusout', (e) =>
  setTimeout(() => searchWrapper.classList.remove('active'), 200)
);
