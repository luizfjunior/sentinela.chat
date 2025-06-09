
import { useState, useRef, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useSupabaseConversations } from "@/hooks/useSupabaseConversations";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import Sidebar from "../components/Sidebar";
import UserProfile from "../components/UserProfile";
import { toast } from "sonner";
import { useSidebar } from "../hooks/useSidebar";

interface LocalMessage {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

const Index = () => {
  const { user } = useAuth();
  const [localMessages, setLocalMessages] = useState<LocalMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [lastGeneratedMessageId, setLastGeneratedMessageId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sidebar = useSidebar();
  
  const {
    conversations,
    currentConversationId,
    messages: supabaseMessages,
    createNewConversation,
    deleteConversation,
    saveMessage,
    updateConversationTitle,
    setCurrentConversationId
  } = useSupabaseConversations();

  // Combine local messages with supabase messages for display
  const allMessages = [...supabaseMessages.map(msg => ({
    id: msg.id,
    content: msg.content,
    role: msg.role as "user" | "assistant",
    timestamp: new Date(msg.created_at)
  })), ...localMessages].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [allMessages]);

  const handleNewChat = () => {
    setCurrentConversationId(null);
    setLocalMessages([]);
    setLastGeneratedMessageId(null);
    sidebar.close();
  };

  const handleSelectConversation = (id: string) => {
    setCurrentConversationId(id);
    setLocalMessages([]);
    setLastGeneratedMessageId(null);
    sidebar.close();
  };

  const handleDeleteConversation = (id: string) => {
    deleteConversation(id);
  };

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || !user) return;

    // Create or get conversation
    let convId = currentConversationId;
    if (!convId) {
      convId = await createNewConversation(content);
      if (!convId) return;
    }

    // Add user message to local state immediately
    const userMessage: LocalMessage = {
      id: `user-${Date.now()}`,
      content,
      role: "user",
      timestamp: new Date()
    };
    setLocalMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      // Save user message to database
      await saveMessage(convId, content, 'user');

      // Update conversation title if this is the first message
      if (supabaseMessages.length === 0 && localMessages.length === 0) {
        const title = content.split(' ').slice(0, 3).join(' ') + '...';
        await updateConversationTitle(convId, title);
      }

      // Send message to API
      const response = await fetch("https://pmogrupooscar.app.n8n.cloud/webhook/chat-sentinela-pd1245", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content })
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

      // Add assistant message to local state
      const assistantMessageId = `assistant-${Date.now()}`;
      const assistantMessage: LocalMessage = {
        id: assistantMessageId,
        content: assistantContent,
        role: "assistant",
        timestamp: new Date()
      };
      
      setLocalMessages(prev => [...prev, assistantMessage]);
      setLastGeneratedMessageId(assistantMessageId);

      // Save assistant message to database
      await saveMessage(convId, assistantContent, 'assistant');
    } catch (error) {
      console.error("Error sending message:", error);
      toast.error("Failed to send message. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleSendAudio = async (audioBlob: Blob) => {
    if (!user) return;
    console.log("Sending audio:", audioBlob);

    // Create or get conversation
    let convId = currentConversationId;
    if (!convId) {
      convId = await createNewConversation("🎵 Áudio enviado");
      if (!convId) return;
    }

    // Add user message indicating audio was sent
    const userMessage: LocalMessage = {
      id: `user-${Date.now()}`,
      content: "🎵 Áudio enviado",
      role: "user",
      timestamp: new Date()
    };
    setLocalMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      // Save user message to database
      await saveMessage(convId, "🎵 Áudio enviado", 'user');

      // Convert blob to base64
      const reader = new FileReader();
      reader.onload = async () => {
        const base64Audio = reader.result as string;
        const response = await fetch("https://pmogrupooscar.app.n8n.cloud/webhook/chat-sentinela-pd1245", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
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

        const assistantMessageId = `assistant-${Date.now()}`;
        const assistantMessage: LocalMessage = {
          id: assistantMessageId,
          content: assistantContent,
          role: "assistant",
          timestamp: new Date()
        };
        
        setLocalMessages(prev => [...prev, assistantMessage]);
        setLastGeneratedMessageId(assistantMessageId);

        // Save assistant message to database
        await saveMessage(convId!, assistantContent, 'assistant');
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
        <div className="max-w-4xl mx-auto flex items-center justify-between px-0 py-[13px]">
          <div className="flex items-center gap-3 ml-12">
            <img 
              alt="Grupo Oscar Logo" 
              src="/lovable-uploads/8d123358-879a-4bd0-8c59-94020f57ed0c.jpg" 
              className="mix-blend-screen w-24 h-auto mx-auto object-fill" 
            />
          </div>
          <UserProfile />
        </div>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto bg-zinc-900">
        <div className="max-w-4xl mx-auto px-4">
          {allMessages.length === 0 ? (
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
              {allMessages.map(message => (
                <ChatMessage 
                  key={message.id} 
                  message={message} 
                  isNewMessage={message.id === lastGeneratedMessageId}
                />
              ))}
              {loading && (
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full overflow-hidden flex items-center justify-center flex-shrink-0">
                    <img 
                      src="/lovable-uploads/iconeIA.jpg" 
                      alt="IA" 
                      className="w-full h-full object-cover"
                    />
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
