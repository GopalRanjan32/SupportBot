const chat = document.getElementById("chat");

const questionInput =
    document.getElementById("questionInput");

const sendButton =
    document.getElementById("sendButton");


function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
}


function addUserMessage(text) {
    const message = document.createElement("div");

    message.className =
        "message user-message";

    message.innerHTML = `
        <div class="message-label">
            You
        </div>

        <div class="message-content"></div>
    `;

    message.querySelector(
        ".message-content"
    ).textContent = text;

    chat.appendChild(message);

    scrollToBottom();
}


function addLoadingMessage() {
    const message = document.createElement("div");

    message.className =
        "message bot-message loading";

    message.id = "loadingMessage";

    message.textContent =
        "SupportBot is searching the docs...";

    chat.appendChild(message);

    scrollToBottom();
}


function removeLoadingMessage() {
    const loadingMessage =
        document.getElementById(
            "loadingMessage"
        );

    if (loadingMessage) {
        loadingMessage.remove();
    }
}


function addBotMessage(data) {
    const message =
        document.createElement("div");

    message.className =
        "message bot-message";

    if (data.refused) {
        message.classList.add(
            "refusal"
        );
    }

    const label =
        document.createElement("div");

    label.className =
        "message-label";

    label.textContent =
        "SupportBot";


    const content =
        document.createElement("div");

    content.className =
        "message-content";

    content.textContent =
        data.answer;


    message.appendChild(label);
    message.appendChild(content);


    if (
        data.citations &&
        data.citations.length > 0
    ) {
        const sources =
            document.createElement("div");

        sources.className =
            "sources";


        const sourcesTitle =
            document.createElement("div");

        sourcesTitle.className =
            "sources-title";

        sourcesTitle.textContent =
            "Sources";


        sources.appendChild(
            sourcesTitle
        );


        data.citations.forEach(
            citation => {

                const source =
                    document.createElement(
                        "div"
                    );

                source.className =
                    "source-item";

                source.textContent =
                    `${citation.source} — chunk ${citation.chunk_number}`;

                sources.appendChild(
                    source
                );
            }
        );


        message.appendChild(
            sources
        );
    }


    chat.appendChild(message);

    scrollToBottom();
}


function addErrorMessage() {
    const message =
        document.createElement("div");

    message.className =
        "message bot-message error-message";

    message.textContent =
        "Something went wrong. Please try again.";

    chat.appendChild(message);

    scrollToBottom();
}


async function askQuestion() {

    const question =
        questionInput.value.trim();


    if (!question) {
        return;
    }


    addUserMessage(question);

    questionInput.value = "";

    sendButton.disabled = true;

    addLoadingMessage();


    try {

        const response =
            await fetch(
                "/ask",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        question: question
                    })
                }
            );


        removeLoadingMessage();


        if (!response.ok) {
            throw new Error(
                `HTTP error: ${response.status}`
            );
        }


        const data =
            await response.json();


        addBotMessage(data);

    } catch (error) {

        console.error(error);

        removeLoadingMessage();

        addErrorMessage();

    } finally {

        sendButton.disabled = false;

        questionInput.focus();
    }
}


sendButton.addEventListener(
    "click",
    askQuestion
);


questionInput.addEventListener(
    "keydown",
    function (event) {

        if (event.key === "Enter") {
            askQuestion();
        }
    }
);