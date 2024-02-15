import React, { useState } from 'react';

function ChatBot() {
    const [chatHistory, setChatHistory] = useState([]);
    const [userInput, setUserInput] = useState('');

    const toggleChat = () => {
        const chatContainer = document.getElementById('chat-container');
        chatContainer.style.display = chatContainer.style.display === 'none' ? 'block' : 'none';
    }

    const sendMessage = () => {
        displayMessage('user', userInput);

        fetch('http://127.0.0.1:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `prompt_school=${encodeURIComponent(userInput)}`,
        })
        .then(response => response.json())
        .then(data => {

          console.log("Data: ",data)
            setTimeout(() => {
                displayMessage('bot', data.bot_response_school);
            }, 1000);
        })
        .catch(error => {
            console.error('Error:', error);
        });

        setUserInput('');
    }

    const displayMessage = (role, message) => {
        setChatHistory(prevHistory => [...prevHistory, { role, content: message }]);
    }

    return (
        <div>
            <button className="chat-btn" onClick={toggleChat}>
                <i className="fas fa-comment"></i>
            </button>

            <div id="chat-container" style={{ display: 'none' }}>
                <div className="message-container">
                    {chatHistory.map((message, index) => (
                        <p key={index} className={`${message.role}-message d-flex`}>
                            {message.role === 'User' ? (
                                <i className="fa-regular fa-user me-2"></i>
                            ) : (
                                <i className="fa-solid fa-robot me-2"></i>
                            )}
                            <span>{message.role}: {message.content}</span>
                        </p>
                    ))}
                </div>
                <form id="chat-form">
                    <div className="input-group mb-3">
                        <input type="text" id="user-input" className="form-control"
                            placeholder="Type your school-related question here" required
                            value={userInput} onChange={e => setUserInput(e.target.value)} />
                        <button type="button" className="btn btn-primary" onClick={sendMessage}>Send</button>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default ChatBot;
