

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { UserRound } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}
interface ChatMessageProps {
  message: Message;
  isNewMessage?: boolean;
}

const ChatMessage = ({
  message,
  isNewMessage = false
}: ChatMessageProps) => {
  const [displayedContent, setDisplayedContent] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isThinking, setIsThinking] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const isUser = message.role === "user";

  // Process content to handle line breaks - let remark-gfm handle table parsing
  const processContent = (content: string) => {
    // Simply replace escaped newlines with real newlines
    return content.replace(/\\n/g, '\n');
  };
  const processedContent = processContent(message.content);

  // Typing animation for new assistant messages only
  useEffect(() => {
    if (isUser || !isNewMessage) {
      setDisplayedContent(processedContent);
      return;
    }

    setIsThinking(true);
    setIsTyping(false);
    setDisplayedContent("");
    setCurrentIndex(0);

    // Show thinking dots for 200ms
    const thinkingTimer = setTimeout(() => {
      setIsThinking(false);
      setIsTyping(true);
    }, 200);
    return () => clearTimeout(thinkingTimer);
  }, [message, isUser, processedContent, isNewMessage]);

  // Handle typing animation
  useEffect(() => {
    if (!isTyping || isUser || !isNewMessage) return;
    if (currentIndex < processedContent.length) {
      const char = processedContent[currentIndex];
      // Variable speed: slower for punctuation, faster for spaces
      const delay = char === '.' || char === '!' || char === '?' ? 100 : char === ',' || char === ';' ? 80 : char === ' ' ? 10 : 20;
      const timeoutId = setTimeout(() => {
        setDisplayedContent(processedContent.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, delay);
      return () => clearTimeout(timeoutId);
    } else {
      setIsTyping(false);
    }
  }, [currentIndex, isTyping, isUser, processedContent, isNewMessage]);

  // Condition for showing AI icon: show when there's actual content to display
  const shouldShowAIIcon = !isUser && (!isNewMessage || isThinking || isTyping || displayedContent.length > 0);

  return (
    <div className={cn("group relative", isUser ? "ml-auto max-w-[80%]" : "mr-auto max-w-full")}>
      <div className={cn("flex gap-4", isUser ? "justify-end" : "justify-start")}>
        {/* Avatar da IA - só aparece quando há conteúdo para mostrar */}
        {shouldShowAIIcon && (
          <div className="w-8 h-8 rounded-full overflow-hidden flex items-center justify-center flex-shrink-0">
            <img
              src="/lovable-uploads/iconeIA.jpg"
              alt="IA"
              className="w-full h-full object-cover"
              loading="eager"
            />
          </div>
        )}

        <div className="max-w-full break-words overflow-hidden">
          {/* Usuário: mensagem pura */}
          {isUser ? (
            <p className="whitespace-pre-wrap leading-relaxed text-white">{displayedContent}</p>
          ) : (
            <div className="markdown-content text-white">
              {/* Always show thinking dots WHEN isThinking */}
              {isThinking ? (
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '300ms' }} />
                </div>
              ) : (
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={{
                  p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed text-white">{children}</p>,
                  h1: ({ children }) => <h1 className="text-xl font-bold mb-3 text-white">{children}</h1>,
                  h2: ({ children }) => <h2 className="text-lg font-bold mb-2 text-white">{children}</h2>,
                  h3: ({ children }) => <h3 className="text-base font-bold mb-2 text-white">{children}</h3>,
                  strong: ({ children }) => <strong className="font-bold text-white">{children}</strong>,
                  em: ({ children }) => <em className="italic text-white">{children}</em>,
                  ol: ({ children }) => <ol className="list-decimal list-inside mb-3 ml-4 text-white">{children}</ol>,
                  ul: ({ children }) => <ul className="list-disc list-inside mb-3 ml-4 text-white">{children}</ul>,
                  li: ({ children }) => <li className="leading-relaxed text-white">{children}</li>,
                  code: ({ children }) => <code className="bg-gray-700 px-1 py-0.5 rounded text-sm font-mono text-white">{children}</code>,
                  pre: ({ children }) => <pre className="bg-gray-700 p-3 rounded-lg overflow-x-auto mb-3 text-white">{children}</pre>,
                  blockquote: ({ children }) => <blockquote className="border-l-4 border-gray-600 pl-4 italic mb-3 text-white">{children}</blockquote>,
                  table: ({ children }) => (
                    <div className="my-4 overflow-x-auto">
                      <table className="w-full border-collapse bg-gray-800 rounded-lg overflow-hidden">
                        {children}
                      </table>
                    </div>
                  ),
                  thead: ({ children }) => <thead className="bg-gray-700">{children}</thead>,
                  tbody: ({ children }) => <tbody>{children}</tbody>,
                  tr: ({ children }) => <tr className="border-b border-gray-600 hover:bg-gray-750">{children}</tr>,
                  th: ({ children }) => (
                    <th className="px-4 py-3 text-left font-semibold text-gray-200 bg-gray-700 border-r border-gray-600 last:border-r-0">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="px-4 py-3 text-gray-100 border-r border-gray-600 last:border-r-0">
                      {children}
                    </td>
                  )
                }}>
                  {displayedContent}
                </ReactMarkdown>
              )}
            </div>
          )}
        </div>

        {/* Avatar do usuário à direita, sem mudanças */}
        {isUser && (
          <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-zinc-600">
            <UserRound className="h-4 w-4 text-white" />
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
