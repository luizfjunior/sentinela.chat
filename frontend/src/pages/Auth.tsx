
import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';

const Auth = () => {
  const {
    user,
    signIn,
    signUp,
    resetPassword,
    loading
  } = useAuth();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [resetEmail, setResetEmail] = useState('');
  const [showResetDialog, setShowResetDialog] = useState(false);

  // Redirect if already authenticated
  useEffect(() => {
    if (user && !loading) {
      navigate('/');
    }
  }, [user, loading, navigate]);

  const [loginForm, setLoginForm] = useState({
    email: '',
    password: ''
  });

  const [signupForm, setSignupForm] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    fullName: ''
  });

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!loginForm.email || !loginForm.password) {
      return;
    }
    setIsLoading(true);
    const { error } = await signIn(loginForm.email, loginForm.password);
    setIsLoading(false);
    if (!error) {
      navigate('/');
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!signupForm.email || !signupForm.password) {
      return;
    }
    if (signupForm.password !== signupForm.confirmPassword) {
      alert('As senhas não coincidem');
      return;
    }
    setIsLoading(true);
    const { error } = await signUp(signupForm.email, signupForm.password, signupForm.fullName);
    setIsLoading(false);
    if (!error) {
      // User will be redirected after email confirmation
    }
  };

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!resetEmail) return;
    
    setIsLoading(true);
    const { error } = await resetPassword(resetEmail);
    setIsLoading(false);
    
    if (!error) {
      setShowResetDialog(false);
      setResetEmail('');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-900 flex items-center justify-center">
        <div className="text-white">Carregando...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <img 
            alt="Grupo Oscar Logo" 
            src="/lovable-uploads/02347520-a11f-464a-bc70-ce3235b73f55.jpg" 
            className="mx-auto h-16 w-auto mix-blend-screen" 
          />
        </div>

        <Card className="bg-zinc-800 border-zinc-700">
          <CardHeader>
            <CardTitle className="text-white text-center">Acesso ao Sentinela</CardTitle>
            <CardDescription className="text-zinc-400 text-center">Entre com uma conta autorizada</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="login" className="space-y-4">
              <TabsList className="grid w-full grid-cols-2 bg-zinc-700">
                <TabsTrigger value="login" className="text-white data-[state=active]:bg-zinc-600">
                  Entrar
                </TabsTrigger>
                <TabsTrigger value="signup" className="text-white data-[state=active]:bg-zinc-600">
                  Cadastrar
                </TabsTrigger>
              </TabsList>

              <TabsContent value="login">
                <form onSubmit={handleLogin} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="login-email" className="text-white">Email</Label>
                    <Input 
                      id="login-email" 
                      type="email" 
                      placeholder="seu@email.com" 
                      value={loginForm.email} 
                      onChange={e => setLoginForm(prev => ({
                        ...prev,
                        email: e.target.value
                      }))} 
                      className="bg-zinc-700 border-zinc-600 text-white placeholder:text-zinc-400" 
                      required 
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="login-password" className="text-white">Senha</Label>
                    <Input 
                      id="login-password" 
                      type="password" 
                      placeholder="Sua senha" 
                      value={loginForm.password} 
                      onChange={e => setLoginForm(prev => ({
                        ...prev,
                        password: e.target.value
                      }))} 
                      className="bg-zinc-700 border-zinc-600 text-white placeholder:text-zinc-400" 
                      required 
                    />
                  </div>
                  <Button type="submit" disabled={isLoading} className="w-full bg-[#873131]">
                    {isLoading ? 'Entrando...' : 'Entrar'}
                  </Button>
                  
                  <Dialog open={showResetDialog} onOpenChange={setShowResetDialog}>
                    <DialogTrigger asChild>
                      <Button type="button" variant="link" className="w-full text-zinc-400 hover:text-white">
                        Esqueceu sua senha?
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="bg-zinc-800 border-zinc-700">
                      <DialogHeader>
                        <DialogTitle className="text-white">Recuperar Senha</DialogTitle>
                        <DialogDescription className="text-zinc-400">
                          Digite seu email para receber o link de recuperação
                        </DialogDescription>
                      </DialogHeader>
                      <form onSubmit={handleResetPassword} className="space-y-4">
                        <div className="space-y-2">
                          <Label htmlFor="reset-email" className="text-white">Email</Label>
                          <Input
                            id="reset-email"
                            type="email"
                            placeholder="seu@email.com"
                            value={resetEmail}
                            onChange={(e) => setResetEmail(e.target.value)}
                            className="bg-zinc-700 border-zinc-600 text-white placeholder:text-zinc-400"
                            required
                          />
                        </div>
                        <Button type="submit" disabled={isLoading} className="w-full bg-[#873131]">
                          {isLoading ? 'Enviando...' : 'Enviar Link de Recuperação'}
                        </Button>
                      </form>
                    </DialogContent>
                  </Dialog>
                </form>
              </TabsContent>

              <TabsContent value="signup">
                <form onSubmit={handleSignup} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="signup-name" className="text-white">Nome Completo</Label>
                    <Input 
                      id="signup-name" 
                      type="text" 
                      placeholder="Seu nome completo" 
                      value={signupForm.fullName} 
                      onChange={e => setSignupForm(prev => ({
                        ...prev,
                        fullName: e.target.value
                      }))} 
                      className="bg-zinc-700 border-zinc-600 text-white placeholder:text-zinc-400" 
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="signup-email" className="text-white">Email</Label>
                    <Input 
                      id="signup-email" 
                      type="email" 
                      placeholder="seu@email.com" 
                      value={signupForm.email} 
                      onChange={e => setSignupForm(prev => ({
                        ...prev,
                        email: e.target.value
                      }))} 
                      className="bg-zinc-700 border-zinc-600 text-white placeholder:text-zinc-400" 
                      required 
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="signup-password" className="text-white">Senha</Label>
                    <Input 
                      id="signup-password" 
                      type="password" 
                      placeholder="Sua senha" 
                      value={signupForm.password} 
                      onChange={e => setSignupForm(prev => ({
                        ...prev,
                        password: e.target.value
                      }))} 
                      className="bg-zinc-700 border-zinc-600 text-white placeholder:text-zinc-400" 
                      required 
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="signup-confirm" className="text-white">Confirmar Senha</Label>
                    <Input 
                      id="signup-confirm" 
                      type="password" 
                      placeholder="Confirme sua senha" 
                      value={signupForm.confirmPassword} 
                      onChange={e => setSignupForm(prev => ({
                        ...prev,
                        confirmPassword: e.target.value
                      }))} 
                      className="bg-zinc-700 border-zinc-600 text-white placeholder:text-zinc-400" 
                      required 
                    />
                  </div>
                  <Button type="submit" disabled={isLoading} className="w-full bg-[#1d6f95]">
                    {isLoading ? 'Criando conta...' : 'Criar Conta'}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Auth;
