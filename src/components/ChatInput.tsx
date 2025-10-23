
import { useState, FormEvent, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Paperclip, Image, Mic } from "lucide-react";
import { useAudioRecording } from "@/hooks/useAudioRecording";
import AudioRecorder from "./AudioRecorder";
import PromptSuggestions from "./PromptSuggestions";
import { toast } from "sonner";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  onSendAudio?: (audioBlob: Blob) => void;
  onSuggestionClick?: (prompt: string) => void;
  isLoading: boolean;
  showSuggestions?: boolean;
}

const ChatInput = ({
  onSendMessage,
  onSendAudio,
  onSuggestionClick,
  isLoading,
  showSuggestions = false
}: ChatInputProps) => {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const {
    isRecording,
    recordingTime,
    startRecording,
    stopRecording,
    cancelRecording,
    isSupported
  } = useAudioRecording();

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading && !isRecording) {
      onSendMessage(input);
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !isRecording) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleStartRecording = async () => {
    if (!isSupported) {
      toast.error("Gravação de áudio não é suportada neste navegador");
      return;
    }
    await startRecording();
  };

  const handleSendAudio = async () => {
    const audioBlob = await stopRecording();
    if (audioBlob && onSendAudio) {
      onSendAudio(audioBlob);
    }
  };

  const handleCancelRecording = () => {
    cancelRecording();
  };

  const handleSuggestionClick = (prompt: string) => {
    setInput(prompt);
    if (onSuggestionClick) {
      onSuggestionClick(prompt);
    }
    // Focus the textarea after filling
    setTimeout(() => {
      textareaRef.current?.focus();
    }, 0);
  };

  // Show suggestions only when input is empty, not recording, not loading
  const shouldShowSuggestions = showSuggestions && !input.trim() && !isRecording && !isLoading;

  if (isRecording) {
    return <div className="relative">
        <AudioRecorder recordingTime={recordingTime} onCancel={handleCancelRecording} onSend={handleSendAudio} />
        <p className="text-xs text-gray-500 dark:text-gray-400 text-center mt-2">
          Gravando áudio... Pressione o botão vermelho para enviar ou X para cancelar.
        </p>
      </div>;
  }

  return <div className="relative">
      {/* Prompt Suggestions */}
      <PromptSuggestions 
        onSuggestionClick={handleSuggestionClick}
        isVisible={shouldShowSuggestions}
      />
      
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative flex items-center rounded-2xl border border-gray-200 dark:border-gray-700 shadow-lg hover:shadow-xl transition-shadow bg-gray-200">
          {/* Attachment buttons */}
          <div className="flex items-center gap-1 p-3">
            <Button type="button" variant="ghost" size="icon" className="h-8 w-8 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700" disabled={isLoading}>
              <Paperclip className="h-4 w-4" />
              <span className="sr-only">Anexar arquivo</span>
            </Button>
            <Button type="button" variant="ghost" size="icon" className="h-8 w-8 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700" disabled={isLoading}>
              <Image className="h-4 w-4" />
              <span className="sr-only">Adicionar imagem</span>
            </Button>
          </div>

          {/* Text input */}
          <Textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown} placeholder="Digite sua mensagem..." className="flex-1 min-h-[20px] max-h-[200px] resize-none border-0 bg-transparent px-0 py-3 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus-visible:ring-0 focus-visible:ring-offset-0" disabled={isLoading} rows={1} />

          {/* Send/Voice buttons */}
          <div className="flex items-center gap-1 p-3">
            {input.trim() ? <Button type="submit" size="icon" className="h-8 w-8 bg-blue-600 hover:bg-blue-700 text-white rounded-lg" disabled={isLoading || !input.trim()}>
                <Send className="h-4 w-4" />
                <span className="sr-only">Enviar mensagem</span>
              </Button> : <Button type="button" variant="ghost" size="icon" className="h-8 w-8 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700" disabled={isLoading || !isSupported} onClick={handleStartRecording}>
                <Mic className="h-4 w-4" />
                <span className="sr-only">Gravar áudio</span>
              </Button>}
          </div>
        </div>
      </form>
      
      {/* Info text */}
      <p className="text-xs text-center mt-2 text-slate-100">
        O Chat Sentinela está em fase de teste, pode cometer erros. Considere verificar informações importantes.
      </p>
    </div>;
};

export default ChatInput;
