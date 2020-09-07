function submitForm (event){
    event.stopImmediatePropagation();
    event.preventDefault();
    var companyName = document.getElementById('company_name').value;

    $.post(
        "/predict",
        { param: companyName },
        function (result) {
            result = JSON.parse(result);
            document.getElementById('probability').innerHTML = "";
            document.getElementById('score').innerHTML = "";
            document.getElementById('score').innerHTML = result.score;
            document.getElementById('probability').innerHTML = result.proba;
        }
    )
}