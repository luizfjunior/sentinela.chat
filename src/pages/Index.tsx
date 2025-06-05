
import { useState, useRef, useEffect } from "react";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import Sidebar from "../components/Sidebar";
import UserProfile from "../components/UserProfile";
import { toast } from "sonner";
import { useSidebar } from "../hooks/useSidebar";
import { useConversations } from "../hooks/useConversations";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

const Index = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const sidebar = useSidebar();
  const {
    conversations,
    currentConversationId,
    createNewConversation,
    deleteConversation,
    updateConversationTitle,
    setCurrentConversationId,
  } = useConversations();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth"
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleNewChat = () => {
    const newConvId = createNewConversation();
    setMessages([]);
    sidebar.close();
  };

  const handleSelectConversation = (id: string) => {
    setCurrentConversationId(id);
    setMessages([]); // In a real app, load messages from the conversation
    sidebar.close();
  };

  const handleDeleteConversation = (id: string) => {
    deleteConversation(id);
    if (currentConversationId === id) {
      setMessages([]);
    }
  };

  const handleLogout = () => {
    toast.success("Sessão terminada com sucesso!");
    // Add logout logic here
  };

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    // Create new conversation if none exists
    let convId = currentConversationId;
    if (!convId) {
      convId = createNewConversation();
    }

    // Update conversation title with first message
    if (messages.length === 0) {
      updateConversationTitle(convId, content);
    }

    // Add user message to chat
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content,
      role: "user",
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await fetch("https://pmogrupooscar.app.n8n.cloud/webhook/chat-sentinela-pd1245", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: content
        })
      });

      const responseText = await response.text();
      let assistantContent = "";

      try {
        if (responseText && responseText.trim()) {
          const data = JSON.parse(responseText);
          assistantContent = data.output || responseText;
        } else {
          assistantContent = "Empty response from server.";
        }
      } catch (parseError) {
        console.log("Response is not JSON, using as plain text:", responseText);
        assistantContent = responseText || "Unknown response format.";
      }

      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        content: assistantContent,
        role: "assistant",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      toast.error("Failed to send message. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleSendAudio = async (audioBlob: Blob) => {
    console.log("Sending audio:", audioBlob);

    // Create new conversation if none exists
    let convId = currentConversationId;
    if (!convId) {
      convId = createNewConversation();
    }

    // Add user message indicating audio was sent
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: "🎵 Áudio enviado",
      role: "user",
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      // Convert blob to base64
      const reader = new FileReader();
      reader.onload = async () => {
        const base64Audio = reader.result as string;
        const response = await fetch("https://pmogrupooscar.app.n8n.cloud/webhook/chat-sentinela-pd1245", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            audioData: base64Audio,
            messageType: "audio"
          })
        });

        const responseText = await response.text();
        let assistantContent = "";

        try {
          if (responseText && responseText.trim()) {
            const data = JSON.parse(responseText);
            assistantContent = data.output || responseText;
          } else {
            assistantContent = "Áudio processado com sucesso.";
          }
        } catch (parseError) {
          console.log("Response is not JSON, using as plain text:", responseText);
          assistantContent = responseText || "Áudio recebido e processado.";
        }

        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          content: assistantContent,
          role: "assistant",
          timestamp: new Date()
        };
        setMessages(prev => [...prev, assistantMessage]);
      };
      reader.readAsDataURL(audioBlob);
    } catch (error) {
      console.error("Error sending audio:", error);
      toast.error("Falha ao enviar áudio. Tente novamente.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 dark:bg-[#0f1218] text-gray-900 dark:text-white">
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebar.isOpen}
        onToggle={sidebar.toggle}
        conversations={conversations}
        currentConversationId={currentConversationId}
        onNewChat={handleNewChat}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
      />

      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur-md border-b border-gray-200 dark:border-gray-800 bg-zinc-700">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3 ml-12">
            <img 
              src="/lovable-uploads/520fc95c-e051-4f07-aa0e-f271a3ba3386.png" 
              alt="Grupo Oscar Logo" 
              className="h-6 w-auto" 
            />
            <span className="text-sm font-medium text-zinc-300">GRUPO OSCAR</span>
          </div>
          
          <UserProfile onLogout={handleLogout} />
        </div>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto bg-zinc-900">
        <div className="max-w-4xl mx-auto px-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center min-h-[60vh]">
              <div className="text-center max-w-xl mx-auto space-y-6 px-4">
                <div className="p-4 rounded-2xl bg-gradient-to-br from-blue-500/10 to-purple-600/10 border border-blue-200/20 dark:border-purple-500/20 bg-zinc-800">
                  <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text mb-3 text-slate-200">
                    Bem-vindo ao Agente Anti-Fraude!
                  </h2>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-6 space-y-6">
              {messages.map(message => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {loading && (
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-medium text-white">AI</span>
                  </div>
                  <div className="typing-indicator mt-1">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Chat Input */}
      <div className="sticky bottom-0 backdrop-blur-md border-t border-gray-200 dark:border-gray-800 bg-zinc-700">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <ChatInput onSendMessage={handleSendMessage} onSendAudio={handleSendAudio} isLoading={loading} />
        </div>
      </div>
    </div>
  );
};

export default Index;
