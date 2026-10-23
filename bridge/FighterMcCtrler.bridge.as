package net.play5d.game.obvn.fighter.ctrler
{
   import flash.display.DisplayObject;
   import flash.display.MovieClip;
   import flash.geom.Point;
   import flash.geom.Rectangle;
   import net.play5d.game.obvn.ctrler.EffectCtrler;
   import net.play5d.game.obvn.ctrler.GameLogic;
   import net.play5d.game.obvn.ctrler.game_ctrler.GameCtrler;
   import net.play5d.game.obvn.data.GameConfig;
   import net.play5d.game.obvn.data.HitType;
   import net.play5d.game.obvn.fighter.Assister;
   import net.play5d.game.obvn.fighter.BaseGameSprite;
   import net.play5d.game.obvn.fighter.FighterAction;
   import net.play5d.game.obvn.fighter.FighterAttacker;
   import net.play5d.game.obvn.fighter.FighterMC;
   import net.play5d.game.obvn.fighter.FighterMain;
   import net.play5d.game.obvn.fighter.data.FighterActionState;
   import net.play5d.game.obvn.fighter.event.FighterEvent;
   import net.play5d.game.obvn.fighter.event.FighterEventDispatcher;
   import net.play5d.game.obvn.fighter.vo.HitVO;
   import net.play5d.game.obvn.fighter.vo.MoveTargetParamVO;
   import net.play5d.game.obvn.interfaces.IFighterActionCtrler;
   import net.play5d.game.obvn.interfaces.IGameSprite;
   
   public class FighterMcCtrler
   {
      
      public var effectCtrler:FighterEffectCtrler;
      
      private var _actionCtrler:IFighterActionCtrler;
      
      private var _mc:FighterMC;
      
      private var _fighter:FighterMain;
      
      private var _assisterCache:Assister;
      
      private var _targetAssisterCache:Assister;
      
      private var _action:FighterAction = new FighterAction();
      
      private var _doingAction:String;
      
      private var _doingAirAction:String;
      
      private var _isFalling:Boolean;
      
      private var _jumpDelayFrame:int = 0;
      
      private var _hurtHoldFrame:int = 0;
      
      private var _defenseHoldFrame:int = 0;
      
      private var _beHitGap:int;
      
      private var _doActionFrame:int;
      
      private var _isTouchFloor:Boolean = true;
      
      private var _isDefense:Boolean;
      
      private var _defenseFrameDelay:int = 0;
      
      private var _moveTargetParam:MoveTargetParamVO;
      
      private var _hurtDownFrame:int;
      
      private var _ghostStepIng:Boolean;
      
      private var _ghostStepFrame:int;
      
      private var _autoDirectFrame:int;
      
      private var _justDefenseFrame:int;
      
      private var _ghostType:int = 0;
      
      private var _justHurtResume:Boolean;
      
      private var _isContinuousGhostStep:Boolean = true;
      
      private var _hurtBreakFrame:uint = 0;
      
      public function FighterMcCtrler(param1:FighterMain)
      {
         super();
         _fighter = param1;
      }
      
      public function destory() : void
      {
         if(_actionCtrler)
         {
            _actionCtrler.destory();
            _actionCtrler = null;
         }
         if(_mc)
         {
            _mc.destory();
            _mc = null;
         }
         _fighter = null;
         _action = null;
         _moveTargetParam = null;
         effectCtrler = null;
         _assisterCache = null;
         _targetAssisterCache = null;
      }
      
      public function getAction() : FighterAction
      {
         return _action;
      }
      
      public function getFighterMc() : FighterMC
      {
         return _mc;
      }
      
      public function getCurAction() : String
      {
         if(_doingAirAction != null)
         {
            return _doingAirAction;
         }
         return _doingAction;
      }
      
      public function setActionCtrler(param1:IFighterActionCtrler) : void
      {
         _actionCtrler = param1;
      }
      
      // ===== 桥接用：暴露动作控制器（注入代码，非原版）=====
      // 桥接靠它判断【这一方是键盘还是 AI】：
      //   fighter.getCtrler().getMcCtrl().getActionCtrler() is FighterAICtrler
      public function getActionCtrler() : IFighterActionCtrler
      {
         return _actionCtrler;
      }

      
      public function setMc(param1:FighterMC) : void
      {
         _mc = param1;
         idle();
      }
      
      public function setSteelBody(param1:Boolean, param2:Boolean = false) : void
      {
         _fighter.isSteelBody = param1;
         _fighter.isSuperSteelBody = param1 && param2;
         if(param1)
         {
            effectCtrler.startGlow(param2 ? 16776960 : 16777215);
         }
         else
         {
            effectCtrler.endGlow();
         }
      }
      
      public function addQi(param1:Number) : void
      {
         _fighter.addQi(param1);
      }
      
      public function idle(param1:String = "站立") : void
      {
         var _loc2_:Boolean = false;
         if(FighterActionState.isHurting(_fighter.actionState))
         {
            _justHurtResume = true;
         }
         _mc.stopHurtFly();
         endAct();
         _doingAction = null;
         _doingAirAction = null;
         setSteelBody(false);
         _justDefenseFrame = 0.1 * GameConfig.FPS_GAME;
         effectCtrler.endShadow();
         effectCtrler.endShake();
         _action.clearAction();
         _action.clearState();
         _fighter.actionState = 0;
         _fighter.isAllowBeHit = !_justHurtResume;
         _fighter.isApplyG = true;
         _fighter.isCross = false;
         _fighter.hurtHit = null;
         _fighter.defenseHit = null;
         _fighter.clearHurtHits();
         _fighter.getDisplay().visible = true;
         _isDefense = false;
         _autoDirectFrame = 0;
         if(!_isTouchFloor && _fighter.isInAir)
         {
            fall();
         }
         else
         {
            _loc2_ = true;
            _fighter.setVelocity(0,0);
            if(param1 == "站立")
            {
               _loc2_ = false;
               _action.jumpTimes = _fighter.jumpTimes;
               _action.airHitTimes = _fighter.airHitTimes;
               setAllAct();
            }
            _mc.goFrame(param1,_loc2_);
         }
      }
      
      public function loop(param1:String) : void
      {
         _mc.goFrame(param1);
      }
      
      public function stop() : void
      {
         _mc.stopRenderMainAnimate();
      }
      
      public function dash(param1:Number = 3) : void
      {
         _action.isDashing = true;
         _fighter.setVelocity(_fighter.speed * param1 * _fighter.direct,0);
         _fighter.setDamping(0,0);
         _fighter.isCross = true;
         _fighter.isAllowBeHit = false;
      }
      
      public function dashStop(param1:Number = 0.5) : void
      {
         var _loc2_:Number = _fighter.getVecX();
         var _loc3_:Number = Math.abs(_loc2_) * param1;
         _fighter.setDamping(_loc3_);
         _fighter.isAllowBeHit = true;
         _fighter.actionState = 0;
         _action.clearAction();
         _action.isDashing = false;
         _fighter.isCross = false;
      }
      
      public function setAllAct() : void
      {
         setMove();
         setDefense();
         setJump();
         setJumpDown();
         setDash();
         setAttack();
         setSkill1();
         setSkill2();
         setZhao1();
         setZhao2();
         setZhao3();
         setCatch1();
         setCatch2();
         setBisha();
         setBishaUP();
         setBishaSUPER();
         setWankai();
      }
      
      public function setAirAllAct() : void
      {
         setDash();
         setAttackAIR();
         setSkillAIR();
         setBishaAIR();
         setAirMove(true);
      }
      
      public function setAirMove(param1:Boolean) : void
      {
         _action.airMove = param1;
      }
      
      public function setMove() : void
      {
         setMoveLeft();
         setMoveRight();
      }
      
      public function setMoveLeft() : void
      {
         _action.moveLeft = "走";
      }
      
      public function setMoveRight() : void
      {
         _action.moveRight = "走";
      }
      
      public function setDefense() : void
      {
         _action.defense = "防御";
      }
      
      public function setJump(param1:String = "跳") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.jump = param1;
      }
      
      public function setJumpQuick(param1:String = "跳") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.jumpQuick = param1;
      }
      
      public function setJumpDown(param1:String = "落") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.jumpDown = param1;
      }
      
      public function setDash(param1:String = "瞬步") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.dash = param1;
      }
      
      public function setAttack(param1:String = "砍1") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.attack = param1;
      }
      
      public function setSkill1(param1:String = "砍技1") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.skill1 = param1;
      }
      
      public function setSkill2(param1:String = "砍技2") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.skill2 = param1;
      }
      
      public function setZhao1(param1:String = "招1") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.zhao1 = param1;
      }
      
      public function setZhao2(param1:String = "招2") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.zhao2 = param1;
      }
      
      public function setZhao3(param1:String = "招3") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.zhao3 = param1;
      }
      
      public function setCatch1(param1:String = "摔1") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.catch1 = param1;
      }
      
      public function setCatch2(param1:String = "摔2") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.catch2 = param1;
      }
      
      public function setBisha(param1:String = "必杀", param2:int = 100) : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.bisha = param1;
         _action.bishaQi = param2;
      }
      
      public function setBishaUP(param1:String = "上必杀", param2:int = 100) : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.bishaUP = param1;
         _action.bishaUPQi = param2;
      }
      
      public function setBishaSUPER(param1:String = "超必杀", param2:int = 300) : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.bishaSUPER = param1;
         _action.bishaSUPERQi = param2;
      }
      
      public function setAttackAIR(param1:String = "跳砍") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.attackAIR = param1;
      }
      
      public function setSkillAIR(param1:String = "跳招") : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.skillAIR = param1;
      }
      
      public function setBishaAIR(param1:String = "空中必杀", param2:int = 100) : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.bishaAIR = param1;
         _action.bishaAIRQi = param2;
      }
      
      public function setTouchFloor(param1:String = "落地", param2:Boolean = true) : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         _action.touchFloor = param1;
         _action.touchFloorBreakAct = param2;
      }
      
      public function setWankai() : void
      {
         if(_mc.checkFrame("万解"))
         {
            _action.wanKai = "万解";
         }
         if(_mc.checkFrame("万解W"))
         {
            _action.wanKaiW = "万解W";
         }
         if(_mc.checkFrame("万解S"))
         {
            _action.wanKaiS = "万解S";
         }
      }
      
      public function setHitTarget(param1:String, param2:String) : void
      {
         _action.hitTarget = param2;
         _action.hitTargetChecker = param1;
      }
      
      public function setHurtAction(param1:String) : void
      {
         _action.hurtAction = param1;
         _fighter.actionState = 16;
      }
      
      public function move(param1:Number = 0, param2:Number = 0) : void
      {
         if(param1 == 0 && param2 == 0)
         {
            stopMove();
            return;
         }
         if(_fighter.isInAir && param1 != 0)
         {
            _action.airMove = false;
         }
         param1 *= _fighter.direct;
         _fighter.setVelocity(param1,param2);
      }
      
      public function movePercent(param1:Number = 0, param2:Number = 0) : void
      {
         move(_fighter.speed * param1,_fighter.speed * param2);
      }
      
      public function stopMove() : void
      {
         _fighter.setVelocity(0,0);
      }
      
      public function damping(param1:Number = 0, param2:Number = 0) : void
      {
         _fighter.setDamping(param1,param2);
      }
      
      public function dampingPercent(param1:Number = 0, param2:Number = 0) : void
      {
         _fighter.setDamping(_fighter.speed * param1,_fighter.speed * param2);
      }
      
      public function endAct() : void
      {
         _action.clearAction();
         _fighter.actionState = 40;
         setSteelBody(false);
         if(_moveTargetParam != null)
         {
            moveTarget(null);
         }
      }
      
      public function fire(param1:String, param2:Object = null) : void
      {
         var mcName:String = param1;
         var params:Object = param2;
         var mc:MovieClip = _mc.getChildByName(mcName) as MovieClip;
         if(mc)
         {
            if(!params)
            {
               params = {};
            }
            params.mc = mc;
            params.hitVO = _fighter.getCtrler().hitModel.getHitVOByDisplayName(mcName);
            FighterEventDispatcher.dispatchEvent(_fighter,"FIRE_BULLET",params);
         }
         else
         {
            _fighter.setAnimateFrameOut(function():void
            {
               fire(mcName,params);
            },1);
         }
      }
      
      public function addAttacker(param1:String, param2:Object = null) : void
      {
         var mcName:String = param1;
         var params:Object = param2;
         var mc:MovieClip = _mc.getChildByName(mcName) as MovieClip;
         if(mc)
         {
            if(!params)
            {
               params = {};
            }
            params.mc = mc;
            params.hitVO = _fighter.getCtrler().hitModel.getHitVOByDisplayName(mcName);
            FighterEventDispatcher.dispatchEvent(_fighter,"ADD_ATTACKER",params);
         }
         else
         {
            _fighter.setAnimateFrameOut(function():void
            {
               addAttacker(mcName,params);
            },1);
         }
      }
      
      public function isApplyG(param1:Boolean) : void
      {
         _fighter.isApplyG = param1;
      }
      
      public function gotoAndPlay(param1:String) : void
      {
         _mc.goFrame(param1,true);
      }
      
      public function gotoAndStop(param1:String) : void
      {
         _mc.goFrame(param1,false);
      }
      
      public function hurtFly(param1:Number, param2:Number) : void
      {
         _mc.playHurtFly(param1 * _fighter.direct,param2,false);
         _action.isHurtFlying = true;
         _fighter.actionState = 22;
         _hurtDownFrame = 0;
         _isFalling = false;
      }
      
      public function moveMC(param1:DisplayObject, param2:Object = null, param3:Object = null) : void
      {
         var _loc4_:IGameSprite = _fighter.getCurrentTarget();
         if(param2)
         {
            if(param2 is Number)
            {
               param1.x = _fighter.x + param2;
            }
            else if(param2.target != undefined && _loc4_)
            {
               param1.x = _loc4_.x - _fighter.x;
               if(isNaN(Number(param2.target)))
               {
                  param1.x += Number(param2.target);
               }
            }
         }
         if(param3)
         {
            if(param3 is Number)
            {
               param1.y = _fighter.y + param3;
            }
            else if(param3.target != undefined && _loc4_)
            {
               param1.y = _loc4_.y - _fighter.y + Number(param3);
               if(isNaN(Number(param3.target)))
               {
                  param1.y += Number(param3.target);
               }
            }
         }
      }
      
      public function justHitToPlay(param1:String, param2:String, param3:Boolean = false, param4:Boolean = false) : void
      {
         if(_fighter.getCtrler().justHit(param1,param4))
         {
            _mc.goFrame(param2);
         }
         else if(param3)
         {
            idle();
         }
      }
      
      public function getAttacker(param1:String) : FighterAttackerCtrler
      {
         var _loc2_:FighterAttacker = GameCtrler.I.getAttacker(param1,_fighter.team.id);
         if(_loc2_)
         {
            return _loc2_.getCtrler();
         }
         return null;
      }
      
      public function getAssister() : Assister
      {
         return GameCtrler.I.getAssister(_fighter.team.id);
      }
      
      public function getTargetAssister() : Assister
      {
         return GameCtrler.I.getAssister(_fighter.team.id == 1 ? 2 : 1);
      }
      
      public function moveTarget(param1:Object = null) : void
      {
         if(!param1 && _moveTargetParam)
         {
            _moveTargetParam.clear();
            _moveTargetParam = null;
            return;
         }
         _moveTargetParam = new MoveTargetParamVO(param1);
         _moveTargetParam.setTarget(_fighter.getCurrentTarget());
      }
      
      public function render() : void
      {
         if(!_assisterCache)
         {
            _assisterCache = getAssister();
         }
         if(!_targetAssisterCache)
         {
            _targetAssisterCache = getTargetAssister();
         }
         if(_ghostStepIng)
         {
            return;
         }
         if(_justDefenseFrame > 0)
         {
            _justDefenseFrame = _justDefenseFrame - 1;
         }
         if(_hurtBreakFrame > 0)
         {
            _hurtBreakFrame = _hurtBreakFrame - 1;
         }
         _action.render();
         if(_moveTargetParam)
         {
            renderMoveTarget();
         }
         if(_actionCtrler)
         {
            _actionCtrler.render();
         }
         if(_action.isHurtFlying)
         {
            renderHurtFlying();
            return;
         }
         if(_action.isHurting)
         {
            renderHurt();
            return;
         }
         if(_action.isDefenseHiting)
         {
            renderDefense(false,true);
            return;
         }
         if(_action.hitTarget)
         {
            renderCheckTargetHit();
         }
         if(renderWanKaiCtrl())
         {
            return;
         }
         if(_fighter && _fighter.isInAir || _doingAirAction && !_action.touchFloorBreakAct)
         {
            renderAirAction();
         }
         else
         {
            renderFloorAction();
         }
      }
      
      private function renderHurtFlying() : void
      {
         if(!_fighter.isInAir)
         {
            _isTouchFloor = true;
         }
         if(!_fighter.isAlive)
         {
            return;
         }
         if(_fighter.actionState == 24)
         {
            _hurtDownFrame = 1;
         }
         if(_hurtDownFrame > 0 && ++_hurtDownFrame < 20)
         {
            if(_actionCtrler.dashJump())
            {
               doHurtDownJump();
               _hurtDownFrame = 0;
            }
         }
      }
      
      private function renderAssist() : void
      {
         if(_fighter.actionState != 0 && _fighter.actionState != 20)
         {
            return;
         }
         if(_actionCtrler.assist() && !_fighter.isUsedPassiveCdIng())
         {
            if(_fighter.fzqi >= 100)
            {
               _fighter.fzqi = 0;
               FighterEventDispatcher.dispatchEvent(_fighter,"ADD_ASSISTER");
            }
         }
      }
      
      private function renderFloorAction() : void
      {
         if(!_isTouchFloor)
         {
            touchFloor();
         }
         if(!_actionCtrler || !_actionCtrler.enabled())
         {
            if(_mc.currentFrameName == "走" || _mc.currentFrameName == "防御")
            {
               idle();
            }
            return;
         }
         renderAssist();
         var _loc2_:IGameSprite = _fighter.getCurrentTarget();
         if(!_loc2_)
         {
            return;
         }
         var _loc3_:Boolean = _fighter.x < _loc2_.x;
         var _loc1_:Boolean = _actionCtrler.moveRIGHT() && _loc3_ && _fighter.direct == 1 || _actionCtrler.moveLEFT() && !_loc3_ && _fighter.direct == -1;
         if(_loc1_)
         {
            if(_action.catch1 && _actionCtrler.catch1())
            {
               doCatch(_action.catch1);
            }
            if(_action.catch2 && _actionCtrler.catch2())
            {
               doCatch(_action.catch2);
            }
         }
         if(_action.bishaSUPER && _actionCtrler.bishaSUPER())
         {
            doBisha(_action.bishaSUPER,_action.bishaSUPERQi,true);
         }
         if(_action.bishaUP && _actionCtrler.bishaUP())
         {
            doBisha(_action.bishaUP,_action.bishaUPQi);
         }
         if(_action.bisha && _actionCtrler.bisha())
         {
            doBisha(_action.bisha,_action.bishaQi);
         }
         if(_action.skill2 && _actionCtrler.skill2())
         {
            doSkill(_action.skill2);
         }
         if(_action.skill1 && _actionCtrler.skill1())
         {
            doSkill(_action.skill1);
         }
         if(_action.zhao3 && _actionCtrler.zhao3())
         {
            doSkill(_action.zhao3);
         }
         if(_action.zhao2 && _actionCtrler.zhao2())
         {
            doSkill(_action.zhao2);
         }
         if(_action.attack && _actionCtrler.attack())
         {
            doAttack(_action.attack);
         }
         if(_action.zhao1 && _actionCtrler.zhao1())
         {
            doSkill(_action.zhao1);
         }
         if(_action.defense && _actionCtrler.defense())
         {
            doDefense();
         }
         if(_action.dash && _actionCtrler.dash())
         {
            doDash(_action.dash);
         }
         if(_action.moveLeft && _actionCtrler.moveLEFT())
         {
            doMove(_action.moveLeft,-1);
         }
         if(_action.moveRight && _actionCtrler.moveRIGHT())
         {
            doMove(_action.moveRight,1);
         }
         if(_action.jump && _actionCtrler.jump())
         {
            doJump(_action.jump);
         }
         if(_action.jumpDown && _actionCtrler.jumpDown())
         {
            doJumpDown(_action.jumpDown);
         }
         if(_action.isMoving)
         {
            renderMoving();
         }
         if(_action.isDefensing)
         {
            renderDefense();
         }
         if(FighterActionState.allowGhostStep(_fighter.actionState))
         {
            if(_actionCtrler.ghostStep())
            {
               doGhostStep();
            }
            if(_actionCtrler.ghostJump())
            {
               doGhostJump();
            }
         }
         if(FighterActionState.isAttacking(_fighter.actionState))
         {
            if(_action.attackAIR && _actionCtrler.attackAIR())
            {
               doAirAttack(_action.attackAIR);
            }
            if(_action.skillAIR && _actionCtrler.skillAIR())
            {
               doAirSkill(_action.skillAIR);
            }
            if(_action.bishaAIR && _actionCtrler.bishaAIR())
            {
               doAirBisha(_action.bishaAIR,_action.bishaAIRQi);
            }
         }
      }
      
      private function renderWanKaiCtrl() : Boolean
      {
         if(!_actionCtrler || !_actionCtrler.enabled())
         {
            return false;
         }
         if(_actionCtrler.wanKaiW())
         {
            return checkDoWankai(_action.wanKaiW,"万解W");
         }
         if(_actionCtrler.wanKaiS())
         {
            return checkDoWankai(_action.wanKaiS,"万解S");
         }
         if(_actionCtrler.wanKai())
         {
            return checkDoWankai(_action.wanKai,"万解");
         }
         return false;
      }
      
      private function checkDoWankai(param1:String, param2:String) : Boolean
      {
         if(param1)
         {
            doWanKaiAction(param1);
            return true;
         }
         if(_mc.currentFrameName == "防御")
         {
            doWanKaiAction(param2);
            return true;
         }
         if(_doingAction == "砍1")
         {
            if(_doActionFrame < 2)
            {
               doWanKaiAction(param2);
               return true;
            }
         }
         return false;
      }
      
      public function renderAnimate() : void
      {
         if(_justHurtResume)
         {
            _fighter.isAllowBeHit = true;
            _justHurtResume = false;
         }
         renderBeHitGap();
         if(_mc)
         {
            _mc.renderAnimate();
         }
         if(_actionCtrler)
         {
            _actionCtrler.renderAnimate();
         }
         if(_ghostStepIng)
         {
            renderGhostStep();
            return;
         }
         if(_action)
         {
            if(_action.isHurting)
            {
               renderHurtAnimate();
            }
            if(_action.isDefenseHiting)
            {
               renderDefensHiting();
            }
            if(_action.isJumping)
            {
               renderJumpAnimate();
            }
            if(_doingAction)
            {
               _doActionFrame = _doActionFrame + 1;
            }
            if(_action.isDefensing)
            {
               renderDefenseAnimate();
            }
         }
         if(_mc && _mc.currentFrameName == "站立")
         {
            if(++_autoDirectFrame > 5)
            {
               _fighter.getCtrler().setDirectToTarget();
               _autoDirectFrame = 0;
            }
         }
      }
      
      private function renderAirAction() : void
      {
         if(!_action.isJumping)
         {
            fall();
         }
         _isTouchFloor = false;
         if(!_actionCtrler || !_actionCtrler.enabled())
         {
            return;
         }
         if(_action.attackAIR && _actionCtrler.attackAIR())
         {
            doAirAttack(_action.attackAIR);
         }
         if(_action.skillAIR && _actionCtrler.skillAIR())
         {
            doAirSkill(_action.skillAIR);
         }
         if(_action.bishaAIR && _actionCtrler.bishaAIR())
         {
            doAirBisha(_action.bishaAIR,_action.bishaAIRQi);
         }
         if(_action.jump && _actionCtrler.jump())
         {
            doAirJump(_action.jump);
         }
         if(_action.jumpQuick && _actionCtrler.jumpQuick())
         {
            doAirJump(_action.jumpQuick);
         }
         if(FighterActionState.isAttacking(_fighter.actionState))
         {
            if(_action.bishaSUPER && _actionCtrler.bishaSUPER())
            {
               doBisha(_action.bishaSUPER,_action.bishaSUPERQi,true);
            }
            if(_action.bishaUP && _actionCtrler.bishaUP())
            {
               doBisha(_action.bishaUP,_action.bishaUPQi);
            }
            if(_action.bisha && _actionCtrler.bisha())
            {
               doBisha(_action.bisha,_action.bishaQi);
            }
            if(_action.skill2 && _actionCtrler.skill2())
            {
               doSkill(_action.skill2);
            }
            if(_action.skill1 && _actionCtrler.skill1())
            {
               doSkill(_action.skill1);
            }
            if(_action.zhao3 && _actionCtrler.zhao3())
            {
               doSkill(_action.zhao3);
            }
            if(_action.zhao2 && _actionCtrler.zhao2())
            {
               doSkill(_action.zhao2);
            }
            if(_action.attack && _actionCtrler.attack())
            {
               doAttack(_action.attack);
            }
            if(_action.zhao1 && _actionCtrler.zhao1())
            {
               doSkill(_action.zhao1);
            }
         }
         if(FighterActionState.allowGhostStep(_fighter.actionState))
         {
            if(_actionCtrler.ghostJump())
            {
               doGhostJump();
            }
            if(_actionCtrler.ghostJumpDown())
            {
               doGhostJumpDown();
            }
         }
         if(_action.dash && _actionCtrler.dash())
         {
            doDashAir(_action.dash);
         }
         if(_action.airMove)
         {
            doAirMove();
         }
      }
      
      private function renderMoveTarget() : void
      {
         var _loc3_:Number = NaN;
         var _loc1_:Number = NaN;
         var _loc2_:DisplayObject = null;
         var _loc4_:IGameSprite = _moveTargetParam.target;
         if(!_loc4_)
         {
            return;
         }
         if(_moveTargetParam.followMcName)
         {
            _loc2_ = _mc.getChildByName(_moveTargetParam.followMcName);
            if(!_loc2_)
            {
               return;
            }
            _loc3_ = _fighter.x + _loc2_.x * _fighter.direct;
            _loc1_ = _fighter.y + _loc2_.y;
         }
         else
         {
            if(!isNaN(_moveTargetParam.x))
            {
               _loc3_ = _moveTargetParam.x;
            }
            if(!isNaN(_moveTargetParam.y))
            {
               _loc1_ = _moveTargetParam.y;
            }
         }
         if(_moveTargetParam.speed)
         {
            if(_moveTargetParam.speed.x > 0 && !isNaN(_loc3_))
            {
               if(_loc4_.x > _loc3_ + _moveTargetParam.speed.x)
               {
                  _loc4_.x -= _moveTargetParam.speed.x;
               }
               if(_loc4_.x < _loc3_ - _moveTargetParam.speed.x)
               {
                  _loc4_.x += _moveTargetParam.speed.x;
               }
               if(_loc4_.y > _loc1_ + _moveTargetParam.speed.y)
               {
                  if(_loc4_ is BaseGameSprite)
                  {
                     (_loc4_ as BaseGameSprite).setVecY(-_moveTargetParam.speed.y);
                     (_loc4_ as BaseGameSprite).setDampingY(1);
                  }
                  else
                  {
                     _loc4_.y -= _moveTargetParam.speed.y;
                  }
               }
               if(_loc4_.y < _loc1_ - _moveTargetParam.speed.y)
               {
                  if(_loc4_ is BaseGameSprite)
                  {
                     (_loc4_ as BaseGameSprite).setVecY(_moveTargetParam.speed.y);
                     (_loc4_ as BaseGameSprite).setDampingY(1);
                  }
                  else
                  {
                     _loc4_.y += _moveTargetParam.speed.y;
                  }
               }
            }
         }
         else
         {
            if(!isNaN(_loc3_))
            {
               _loc4_.x = _loc3_;
            }
            if(!isNaN(_loc1_))
            {
               _loc4_.y = _loc1_;
            }
         }
      }
      
      private function fall() : void
      {
         if(_isFalling)
         {
            return;
         }
         if(_doingAction)
         {
            return;
         }
         _action.clearState();
         _action.clearAction();
         setAirAllAct();
         setJump();
         _isFalling = true;
         _doingAirAction = null;
         _isTouchFloor = false;
         _isDefense = false;
         _fighter.setVecX(0);
         setTouchFloor("落地",true);
         _mc.goFrame("落",false);
      }
      
      public function touchFloor() : void
      {
         if(!_fighter.isAlive)
         {
            return;
         }
         var _loc2_:String = _action.touchFloor;
         if(_isFalling)
         {
            if(!_loc2_)
            {
               _loc2_ = "落地";
            }
         }
         if(_loc2_ == null)
         {
            return;
         }
         var _loc1_:Object = _loc2_ == "落地" ? {
            "call":setAttack,
            "delay":1
         } : null;
         doAction(_loc2_,false,_loc1_);
         effectCtrler.touchFloor();
         _action.airHitTimes = _fighter.airHitTimes;
         _action.jumpTimes = _fighter.jumpTimes;
         _isTouchFloor = true;
         _isFalling = false;
      }
      
      private function doAction(param1:String, param2:Boolean = false, param3:Object = null) : void
      {
         if(param1 == null)
         {
            return;
         }
         effectCtrler.endShadow();
         effectCtrler.endShake();
         _fighter.setVelocity(0,0);
         _action.isMoving = false;
         _action.isDefensing = false;
         _action.isDashing = false;
         _doingAction = param1;
         _doingAirAction = param2 ? param1 : null;
         _action.clearAction();
         _isFalling = false;
         _isDefense = false;
         _fighter.isAllowBeHit = true;
         _fighter.isCross = false;
         _fighter.isApplyG = true;
         _doActionFrame = 0;
         _mc.goFrame(param1,true,0,param3);
         _fighter.dispatchEvent(new FighterEvent("DO_ACTION"));
      }
      
      private function setMoveAction() : void
      {
         _action.clearAction();
         _action.isMoving = true;
         setMove();
         setAttack();
         setZhao1();
         setZhao3();
         setSkill2();
         setJump();
         setDash();
         setBisha();
         setBishaUP();
         setDefense();
         setCatch1();
         setCatch2();
      }
      
      private function renderMoving() : void
      {
         if(_actionCtrler.moveLEFT())
         {
            _fighter.direct = -1;
            move(_fighter.speed);
         }
         else if(_actionCtrler.moveRIGHT())
         {
            _fighter.direct = 1;
            move(_fighter.speed);
         }
         else
         {
            idle();
         }
      }
      
      private function doMove(param1:String, param2:int = 1) : void
      {
         if(_action.isMoving)
         {
            return;
         }
         _mc.goFrame(param1,true);
         _fighter.actionState = 0;
         setMoveAction();
      }
      
      private function doAirMove() : void
      {
         if(_actionCtrler.moveLEFT())
         {
            _fighter.move(-_fighter.speed);
         }
         if(_actionCtrler.moveRIGHT())
         {
            _fighter.move(_fighter.speed);
         }
      }
      
      private function renderDefense(param1:Boolean = true, param2:Boolean = false) : void
      {
         if(_actionCtrler.moveLEFT())
         {
            if(_fighter.direct != -1)
            {
               _fighter.direct = -1;
               setDefenseAction(param1,param2);
            }
         }
         if(_actionCtrler.moveRIGHT())
         {
            if(_fighter.direct != 1)
            {
               _fighter.direct = 1;
               setDefenseAction(param1,param2);
            }
         }
      }
      
      private function renderDefenseAnimate() : void
      {
         if(_action.isDefenseHiting)
         {
            return;
         }
         if(_defenseFrameDelay-- > 0)
         {
            return;
         }
         if(_actionCtrler.enabled() && _actionCtrler.defense())
         {
            if(!_isDefense)
            {
               _isDefense = true;
            }
            return;
         }
         if(_defenseFrameDelay > -5)
         {
            return;
         }
         _action.isDefensing = false;
         _mc.goFrame("防御恢复",false,0,{
            "call":idle,
            "delay":1
         });
      }
      
      private function setDefenseAction(param1:Boolean = true, param2:Boolean = false) : void
      {
         if(param1)
         {
            _action.clearAction();
            _action.clearState();
            _action.isDefensing = true;
            setSkill1();
            setZhao2();
            setBishaSUPER();
            setJumpDown();
         }
         if(param2)
         {
            _isDefense = true;
         }
         else
         {
            _isDefense = _justDefenseFrame > 0;
         }
         _defenseFrameDelay = 1;
         _mc.goFrame("防御",true,3);
      }
      
      private function doDefense() : void
      {
         if(_action.isDefensing)
         {
            return;
         }
         _fighter.actionState = 20;
         dampingPercent(1,1);
         setDefenseAction();
      }
      
      private function doDash(param1:String) : void
      {
         if(!_fighter.hasEnergy(20,true))
         {
            return;
         }
         _fighter.useEnergy(20);
         if(_actionCtrler.moveLEFT())
         {
            _fighter.direct = -1;
         }
         if(_actionCtrler.moveRIGHT())
         {
            _fighter.direct = 1;
         }
         doAction(param1);
         _fighter.actionState = 15;
         _fighter.isAllowBeHit = false;
         isApplyG(false);
      }
      
      private function doDashAir(param1:String) : void
      {
         if(_action.jumpTimes < 1)
         {
            return;
         }
         if(!_fighter.hasEnergy(30,true))
         {
            return;
         }
         _fighter.useEnergy(30);
         doAction(param1);
         _fighter.actionState = 15;
         _fighter.isAllowBeHit = false;
         isApplyG(false);
         _action.jumpTimes = 0;
      }
      
      private function doJump(param1:String) : void
      {
         if(_action.jumpTimes <= 0)
         {
            return;
         }
         _action.clearAction();
         _action.clearState();
         _doingAction = null;
         _doingAirAction = null;
         _mc.goFrame("起跳",false);
         _jumpDelayFrame = 2;
         _action.isJumping = true;
         _fighter.actionState = 14;
      }
      
      private function doJumpDown(param1:String) : void
      {
         if(_fighter.isTouchBottom)
         {
            return;
         }
         _action.clear();
         _action.jumpTimes = 0;
         _fighter.setVecY(5);
         _fighter.setDamping(0,1);
         _fighter.y += 1;
         _isDefense = false;
         _mc.goFrame(param1,false);
         setTouchFloor();
      }
      
      private function doAirJump(param1:String) : void
      {
         if(_action.jumpTimes <= 0)
         {
            return;
         }
         _action.clearAction();
         _action.clearState();
         _doingAction = null;
         _doingAirAction = null;
         _jumpDelayFrame = 1;
         _action.isJumping = true;
         _fighter.actionState = 14;
      }
      
      public function renderJumpAnimate() : void
      {
         if(_doingAction)
         {
            return;
         }
         if(_jumpDelayFrame > 0)
         {
            _jumpDelayFrame = _jumpDelayFrame - 1;
            if(_jumpDelayFrame == 0)
            {
               _isFalling = false;
               _action.jumpTimes--;
               _mc.goFrame("跳",false);
               _fighter.jump();
               setAirAllAct();
               if(_fighter.isInAir)
               {
                  effectCtrler.jumpAir();
               }
               else
               {
                  effectCtrler.jump();
               }
            }
            return;
         }
         if(_mc.getCurrentFrameCount() == 2)
         {
            setJumpQuick();
         }
         var _loc1_:Number = _fighter.getVecY();
         if(_mc.currentFrameName == "跳中")
         {
            if(_mc.getCurrentFrameCount() >= 4)
            {
               setJump();
            }
         }
         else if(_loc1_ > -_fighter.jumpPower * 0.35)
         {
            _mc.goFrame("跳中",false);
         }
         if(_loc1_ >= 0)
         {
            _action.isJumping = false;
            _isFalling = true;
         }
      }
      
      private function doAttack(param1:String) : void
      {
         doAction(param1);
         _fighter.actionState = 10;
      }
      
      private function doSkill(param1:String) : void
      {
         doAction(param1);
         _fighter.actionState = 11;
      }
      
      private function doCatch(param1:String) : void
      {
         if(!allowCatch())
         {
            return;
         }
         doAction(param1);
         _fighter.actionState = 11;
      }
      
      private function doBisha(param1:String, param2:int, param3:Boolean = false) : void
      {
         if(!_fighter.useQi(param2))
         {
            return;
         }
         _fighter.actionState = param3 ? 13 : 12;
         doAction(param1);
      }
      
      private function doWanKaiAction(param1:String) : void
      {
         if(!_mc.checkFrame(param1))
         {
            return;
         }
         if(!_fighter.useQi(300))
         {
            return;
         }
         _fighter.actionState = 50;
         doAction(param1);
         _fighter.isAllowBeHit = false;
      }
      
      private function doAirAttack(param1:String) : void
      {
         if(_doingAction == null && _action.airHitTimes <= 0)
         {
            return;
         }
         _fighter.addDamping(0,3);
         _action.airHitTimes--;
         _action.jumpTimes = 0;
         doAction(param1,true);
         _fighter.actionState = 10;
      }
      
      private function doAirSkill(param1:String) : void
      {
         var _loc2_:String = null;
         if(_doingAction == null && _action.airHitTimes <= 0)
         {
            return;
         }
         if(_action.isJumping)
         {
            _loc2_ = null;
            if(_actionCtrler.zhao3())
            {
               _loc2_ = "跳招-上";
            }
            if(_actionCtrler.zhao2())
            {
               _loc2_ = "跳招-下";
            }
            if(_loc2_ && _mc.checkFrame(_loc2_))
            {
               param1 = _loc2_;
            }
         }
         _action.airHitTimes = 0;
         _action.jumpTimes = 0;
         doAction(param1,true);
         _fighter.actionState = 11;
      }
      
      private function doAirBisha(param1:String, param2:int) : void
      {
         if(_doingAction == null && _action.airHitTimes <= 0)
         {
            return;
         }
         if(!_fighter.useQi(param2))
         {
            return;
         }
         _fighter.actionState = 12;
         _action.airHitTimes = 0;
         doAction(param1,true);
      }
      
      public function beHit(param1:HitVO, param2:Rectangle = null) : void
      {
         var _loc6_:BaseGameSprite = null;
         var _loc5_:Number = NaN;
         _loc5_ = 0.3;
         var _loc7_:Boolean = false;
         var _loc3_:int = 0;
         var _loc4_:Number = NaN;
         if(_action.hurtAction)
         {
            doAction(_action.hurtAction);
            return;
         }
         if(_moveTargetParam != null)
         {
            moveTarget(null);
         }
         param1 = settlementBuffBefore(param1);
         if(_fighter.getIsTouchSide())
         {
            if(param1.owner && param1.owner is BaseGameSprite)
            {
               _loc6_ = param1.owner as BaseGameSprite;
               if(Math.abs(_fighter.x - param1.owner.x) < 100)
               {
                  _loc7_ = _fighter.x < _loc6_.x;
                  _loc3_ = _loc7_ ? 1 : -1;
                  if(_loc7_ && _loc6_.direct == 1 || !_loc7_ && _loc6_.direct == -1)
                  {
                     _loc3_ = 0;
                  }
                  _loc4_ = param1.hitx * _loc3_ * 1.4;
                  if(_loc4_ > 20)
                  {
                     _loc4_ = 20;
                  }
                  if(_loc4_ < -20)
                  {
                     _loc4_ = -20;
                  }
                  _loc6_.setVec2(_loc4_,0,_loc5_,0);
               }
            }
         }
         if(_fighter.isSteelBody)
         {
            if(_fighter.isAlive)
            {
               doSteelHurt(param1,param2);
            }
         }
         else if(_isDefense)
         {
            if(param1.isCatch() || param1.isReal)
            {
               if(param1.isReal)
               {
                  trace("该伤害为真实伤害，不可防御：" + param1.id);
               }
               doHurt(param1,param2);
               return;
            }
            if(param1.checkDirect && param1.owner)
            {
               if(checkDefDirect(param1.owner))
               {
                  doHurt(param1,param2);
                  return;
               }
            }
            doDefenseHit(param1,param2);
         }
         else
         {
            doHurt(param1,param2);
         }
      }
      
      private function settlementBuffBefore(param1:HitVO) : HitVO
      {
         var _loc5_:Number = NaN;
         var _loc3_:Number = NaN;
         var _loc2_:HitVO = param1;
         var _loc4_:Object = _assisterCache ? _assisterCache.buffEffects : null;
         if(_loc4_)
         {
            if(_loc4_.attrResistance && HitType.isAttribute(_loc2_.hitType))
            {
               _loc5_ = Number(_loc4_.attrResistance);
               if(_loc5_ > 1)
               {
                  _loc5_ = 1;
               }
               if(_loc5_ < 0)
               {
                  _loc5_ = 0;
               }
               _loc3_ = _loc2_.power * _loc5_;
               _loc2_ = _loc2_.clone();
               _loc2_.power -= _loc3_;
               trace("属性抗性：" + _loc5_ + "，减免伤害：" + _loc3_);
            }
         }
         return _loc2_;
      }
      
      private function settlementBuffAfter(param1:HitVO) : void
      {
         var _loc5_:Number = NaN;
         var _loc4_:Number = NaN;
         if(!_fighter || !_fighter.getCurrentTarget())
         {
            return;
         }
         var _loc3_:FighterMain = _fighter.getCurrentTarget() as FighterMain;
         if(!_loc3_)
         {
            return;
         }
         var _loc2_:Object = _targetAssisterCache ? _targetAssisterCache.buffEffects : null;
         if(!_loc2_)
         {
            return;
         }
         if(_loc2_.phyLifesteal && HitType.isPhysical(param1.hitType) && !param1.isBisha() && FighterActionState.isHurting(_fighter.actionState))
         {
            _loc5_ = Number(_loc2_.phyLifesteal);
            if(_loc5_ > 1)
            {
               _loc5_ = 1;
            }
            if(_loc5_ < 0)
            {
               _loc5_ = 0;
            }
            _loc4_ = param1.getDamage() * _loc5_;
            _loc3_.getCtrler().addHp(_loc4_);
            trace("物理吸血：" + _loc5_ + "，吸取血量：" + _loc4_);
         }
      }
      
      private function checkDefDirect(param1:IGameSprite) : Boolean
      {
         if(_fighter.x < param1.x - 5)
         {
            return _fighter.direct < 0 && param1.direct < 0;
         }
         if(_fighter.x > param1.x + 5)
         {
            return _fighter.direct > 0 && param1.direct > 0;
         }
         return false;
      }
      
      private function doSteelHurt(param1:HitVO, param2:Rectangle) : void
      {
         var _loc6_:Number = NaN;
         var _loc3_:Number = NaN;
         var _loc5_:Number = NaN;
         var _loc4_:Number = NaN;
         if(!_fighter.isSuperSteelBody && (_fighter.energyOverLoad || param1.isBisha() || param1.isCatch()))
         {
            doHurt(param1,param2);
            return;
         }
         _fighter.hurtHit = param1;
         _fighter.loseHp(param1.getDamage() * (_fighter.isSuperSteelBody ? 0.3 : 0.65));
         if(_fighter.isAlive && GameLogic.checkFighterDie(_fighter))
         {
            FighterEventDispatcher.dispatchEvent(_fighter,"DIE");
            _fighter.isAlive = false;
            doHurt(param1,param2);
            return;
         }
         if(param1.hurtType == 1)
         {
            _beHitGap = 10;
         }
         else
         {
            _beHitGap = 4;
         }
         if(_fighter.isSuperSteelBody)
         {
            _fighter.useEnergy(param1.getDamage() * 0.2);
         }
         else if(param1.isBreakDef)
         {
            _fighter.useEnergy(param1.getDamage());
         }
         else
         {
            _fighter.useEnergy(param1.getDamage() * 0.4);
         }
         _fighter.isAllowBeHit = false;
         if(!_fighter.isSuperSteelBody)
         {
            _loc6_ = param1.hitx;
            _loc3_ = param1.hity;
            if(param1.owner)
            {
               _loc6_ *= param1.owner.direct;
            }
            _loc5_ = _loc6_;
            _loc4_ = _loc3_;
            if(param1.isBreakDef)
            {
               _loc5_ *= 2;
               _loc4_ *= 2;
            }
            _fighter.setVec2(_loc5_,_loc4_,Math.abs(_loc6_ * 0.1),Math.abs(_loc3_ * 0.1));
         }
         if(param1 && param2)
         {
            EffectCtrler.I.doSteelHitEffect(param1,param2,_fighter);
         }
      }
      
      private function doHurt(param1:HitVO, param2:Rectangle) : void
      {
         effectCtrler.endShadow();
         effectCtrler.endShake();
         _fighter.hurtHit = param1;
         _fighter.loseHp(param1.getDamage());
         if(_fighter.isAlive && GameLogic.checkFighterDie(_fighter))
         {
            FighterEventDispatcher.dispatchEvent(_fighter,"DIE");
            _fighter.isAlive = false;
         }
         _beHitGap = 4;
         _fighter.isCross = false;
         _fighter.isAllowBeHit = false;
         _fighter.isApplyG = true;
         _isDefense = false;
         var _loc4_:Number = param1.hitx;
         var _loc3_:Number = param1.hity;
         if(param1.owner)
         {
            _loc4_ *= param1.owner.direct;
         }
         if(_fighter.isInAir)
         {
            if(_loc3_ <= 0)
            {
               _loc3_ -= 3;
            }
         }
         else if(_loc3_ < 0)
         {
            _loc3_ -= 6;
            _isTouchFloor = false;
         }
         _action.clearState();
         _doingAirAction = null;
         _doingAction = null;
         setSteelBody(false);
         if(param1.hurtType == 0)
         {
            _action.isHurting = true;
            _hurtHoldFrame = Math.round(param1.hurtTime / 1000 * 30) + GameConfig.HURT_FRAME_OFFSET;
            if(_hurtHoldFrame < 4)
            {
               _hurtHoldFrame = 4;
            }
            if(param1.hitType == 11)
            {
               _mc.goFrame("被打",false);
            }
            else
            {
               _mc.goFrame("被打",true,7);
            }
            _fighter.actionState = 21;
            _fighter.setVelocity(_loc4_,_loc3_);
            _fighter.setDamping(0.1,0.5);
            _fighter.isApplyG = param1.isApplyG;
            if(_fighter.isAlive && HitType.isHeavy(param1.hitType))
            {
               _fighter.getCtrler().getVoiceCtrl().playVoice(0,0.5);
            }
         }
         else
         {
            _action.isHurtFlying = true;
            _fighter.actionState = 22;
            _hurtDownFrame = 0;
            _mc.playHurtFly(_loc4_,_loc3_);
            if(_fighter.isAlive)
            {
               _fighter.getCtrler().getVoiceCtrl().playVoice(1,1);
            }
            else
            {
               _fighter.getCtrler().getVoiceCtrl().playVoice(2,1);
            }
         }
         if(param1 && param2)
         {
            EffectCtrler.I.doHitEffect(param1,param2,_fighter);
         }
         _isFalling = false;
         settlementBuffAfter(param1);
      }
      
      private function renderHurt() : void
      {
         if(!_fighter.isAlive)
         {
            return;
         }
         renderHurtBreak();
      }
      
      public function isCanHurtBreak() : Boolean
      {
         if(_hurtBreakFrame > 0 || !_fighter.hasEnergy(50) || _fighter.qi < 100)
         {
            return false;
         }
         var _loc4_:HitVO = _fighter.getLastHurtHitVO();
         var _loc3_:Boolean = _loc4_ && _loc4_.isBisha();
         var _loc1_:Boolean = _fighter.hurtBreakHit();
         var _loc2_:int = _fighter.currentHurtDamage();
         return !_loc3_ && !_loc1_ && _loc2_ <= 210;
      }
      
      private function renderHurtBreak() : void
      {
         if(!_actionCtrler.specailSkill() || !isCanHurtBreak())
         {
            return;
         }
         _hurtBreakFrame = 5;
         _fighter.useQi(100);
         _fighter.useEnergy(100);
         if(_fighter.data.comicType == 1)
         {
            _fighter.replaceSkill();
         }
         else
         {
            _fighter.energyExplode();
         }
         FighterEventDispatcher.dispatchEvent(_fighter,"HURT_RESUME");
      }
      
      private function renderHurtAnimate() : void
      {
         var _loc1_:Point = null;
         if(_hurtHoldFrame-- <= 0)
         {
            if(!_fighter.isAlive)
            {
               _action.clearState();
               if(_fighter.isInAir)
               {
                  _loc1_ = _fighter.getVec2();
                  hurtFly(_loc1_.x,_loc1_.y);
               }
               else
               {
                  _mc.playHurtDown();
               }
               _fighter.getCtrler().getVoiceCtrl().playVoice(2,1);
            }
            else
            {
               hurtResume();
            }
         }
      }
      
      private function hurtResume() : void
      {
         if(!_fighter.isInAir && !_isTouchFloor)
         {
            _isTouchFloor = true;
         }
         idle();
         FighterEventDispatcher.dispatchEvent(_fighter,"HURT_RESUME");
      }
      
      private function renderBeHitGap() : void
      {
         if(_beHitGap > 0)
         {
            if(--_beHitGap <= 0)
            {
               _fighter.isAllowBeHit = true;
            }
         }
      }
      
      private function doDefenseHit(param1:HitVO, param2:Rectangle) : void
      {
         _fighter.loseHp(param1.getDamage() * 0.05);
         if(_fighter.isAlive && GameLogic.checkFighterDie(_fighter))
         {
            FighterEventDispatcher.dispatchEvent(_fighter,"DIE");
            _fighter.isAlive = false;
            doHurt(param1,param2);
            return;
         }
         _fighter.defenseHit = param1;
         var _loc3_:int = 0;
         if(param1.isBreakDef)
         {
            _loc3_ = _fighter.energyMax * 0.9;
         }
         else
         {
            _loc3_ = param1.getDamage() / 5;
            if(_loc3_ > 50)
            {
               _loc3_ = 50;
            }
         }
         if(!_fighter.hasEnergy(_loc3_,false))
         {
            _fighter.useEnergy(_loc3_);
            doBreakDefense(param1,param2);
            return;
         }
         _fighter.useEnergy(_loc3_);
         _beHitGap = 4;
         _fighter.isAllowBeHit = false;
         var _loc4_:Number = param1.hitx;
         if(param1.owner)
         {
            _loc4_ *= param1.owner.direct;
         }
         _action.isDefenseHiting = true;
         if(param1.hurtType == 0)
         {
            _defenseHoldFrame = param1.hurtTime / 1000 * GameConfig.FPS_GAME / 5;
            if(_defenseHoldFrame < 5)
            {
               _defenseHoldFrame = 5;
            }
            if(_defenseHoldFrame > 10)
            {
               _defenseHoldFrame = 10;
            }
         }
         else
         {
            _defenseHoldFrame = 10;
            _beHitGap = 8;
         }
         _fighter.setVelocity(_loc4_,0);
         _fighter.setDamping(1,0);
         if(param1 && param2)
         {
            EffectCtrler.I.doDefenseEffect(param1,param2,_fighter.defenseType);
         }
      }
      
      private function doBreakDefense(param1:HitVO, param2:Rectangle) : void
      {
         var _loc5_:Number = NaN;
         var _loc3_:Number = NaN;
         _fighter.loseHp(param1.getDamage() / 10);
         if(param1.hurtType == 0)
         {
            _beHitGap = 4;
         }
         if(param1.hurtType == 1)
         {
            _beHitGap = 10;
         }
         _fighter.isAllowBeHit = false;
         _fighter.energyOverLoad = false;
         _isDefense = false;
         var _loc4_:Number = param1.hitx;
         if(_loc4_ < 5)
         {
            _loc4_ = 5;
         }
         if(_loc4_ > 10)
         {
            _loc4_ = 10;
         }
         if(param1.owner)
         {
            _loc4_ *= param1.owner.direct;
         }
         _action.clearState();
         _action.isHurting = true;
         _hurtHoldFrame = 42;
         _mc.goFrame("被打",true,7);
         _fighter.actionState = 21;
         _fighter.setVelocity(_loc4_);
         _fighter.setDamping(0.1);
         if(param1 && param2)
         {
            _loc5_ = param2.x + param2.width / 2;
            _loc3_ = param2.y + param2.height / 2;
            EffectCtrler.I.doDefenseEffect(param1,param2,_fighter.defenseType);
            EffectCtrler.I.doEffectById("break_def",_loc5_,_loc3_,_fighter.direct);
         }
      }
      
      private function renderDefensHiting() : void
      {
         if(_defenseHoldFrame > 0)
         {
            _defenseHoldFrame = _defenseHoldFrame - 1;
         }
         else if(_fighter.getVecX() == 0)
         {
            _action.isDefenseHiting = false;
         }
      }
      
      private function renderCheckTargetHit() : void
      {
         var _loc5_:int = 0;
         var _loc1_:Rectangle = null;
         var _loc2_:String = _action.hitTargetChecker;
         if(!_loc2_)
         {
            return;
         }
         var _loc3_:Rectangle = _fighter.getCtrler().getHitCheckRect(_loc2_);
         if(!_loc3_)
         {
            return;
         }
         var _loc4_:Vector.<IGameSprite> = _fighter.getTargets();
         if(!_loc4_)
         {
            return;
         }
         _loc5_ = 0;
         while(_loc5_ < _loc4_.length)
         {
            if(_loc4_[_loc5_] is FighterMain)
            {
               _loc1_ = _loc4_[_loc5_].getBodyArea();
               if(_loc1_ && _loc3_.intersects(_loc1_))
               {
                  doAction(_action.hitTarget);
               }
            }
            _loc5_++;
         }
      }
      
      private function allowCatch() : Boolean
      {
         var _loc3_:Point = null;
         var _loc5_:IGameSprite = _fighter.getCurrentTarget();
         if(!_loc5_)
         {
            return false;
         }
         if(_loc5_ is FighterMain)
         {
            if((_loc5_ as FighterMain).actionState == 21)
            {
               return false;
            }
         }
         var _loc6_:Rectangle = _loc5_.getBodyArea();
         var _loc4_:Rectangle = _fighter.getBodyArea();
         if(!_loc6_ || !_loc4_)
         {
            return false;
         }
         if(_fighter.isCheckedRender && _fighter.isRenderBeforeTarget)
         {
            _loc3_ = getTargetNextPos();
            if(_loc3_)
            {
               _loc6_.x += _loc3_.x;
               _loc6_.y += _loc3_.y;
            }
         }
         var _loc2_:Number = 0;
         var _loc1_:Number = Math.abs(_fighter.y - _loc5_.y);
         if(_loc4_.x < _loc6_.x)
         {
            if(_fighter.direct < 0)
            {
               return false;
            }
            _loc2_ = _loc6_.x - (_loc4_.x + _loc4_.width);
         }
         else
         {
            if(_fighter.direct > 0)
            {
               return false;
            }
            _loc2_ = _loc4_.x - (_loc6_.x + _loc6_.width);
         }
         return _loc2_ < 2 && _loc1_ < 1;
      }
      
      private function getTargetNextPos() : Point
      {
         if(!_fighter.isCheckedRender)
         {
            return null;
         }
         var _loc1_:IGameSprite = _fighter.getCurrentTarget();
         if(!_loc1_ || !(_loc1_ is BaseGameSprite))
         {
            return null;
         }
         var _loc4_:BaseGameSprite = _loc1_ as BaseGameSprite;
         var _loc3_:Number = 0;
         var _loc5_:Number = 0;
         if(_loc4_.getVecX())
         {
            _loc3_ += _loc4_.getVecX();
         }
         if(_loc4_.getVecY())
         {
            _loc5_ += _loc4_.getVecY();
         }
         var _loc2_:Point = _loc4_.getVec2();
         if(_loc2_.x && _loc2_.y)
         {
            _loc3_ += _loc2_.x;
            _loc5_ += _loc2_.y;
         }
         _loc3_ *= GameConfig.SPEED_PLUS_DEFAULT;
         _loc5_ *= GameConfig.SPEED_PLUS_DEFAULT;
         return new Point(_loc3_,_loc5_);
      }
      
      private function doHurtDownJump() : void
      {
         if(_doingAction == "起身")
         {
            return;
         }
         if(_fighter.currentHurtDamage() > 240)
         {
            return;
         }
         if(!_fighter.hasEnergy(30))
         {
            return;
         }
         _mc.stopHurtFly();
         _fighter.useEnergy(30);
         var _loc1_:Number = _fighter.getVecX();
         doAction("起身");
         _fighter.isAllowBeHit = false;
         _fighter.setVelocity(_loc1_);
         _fighter.setDamping(_loc1_ * 0.1);
         FighterEventDispatcher.dispatchEvent(_fighter,"HURT_RESUME");
      }
      
      public function sayIntro() : void
      {
         _fighter.actionState = 60;
         _mc.goFrame("开场");
      }
      
      public function doWin() : void
      {
         _fighter.actionState = 61;
         _mc.goFrame("胜利");
      }
      
      public function doLose() : void
      {
         _fighter.actionState = 62;
         _mc.goFrame("失败");
      }
      
      private function doGhostStep() : void
      {
         if(_isContinuousGhostStep && startGhostStep())
         {
            move(8,0);
            _mc.goFrame("走",true);
            _ghostType = 0;
         }
      }
      
      private function doGhostJump() : void
      {
         if(startGhostStep())
         {
            move(0,-12);
            damping(0,0.1);
            _mc.goFrame("跳",false);
            _action.jumpTimes--;
            _ghostType = 1;
         }
      }
      
      private function doGhostJumpDown() : void
      {
         if(startGhostStep())
         {
            move(0,15);
            _mc.goFrame("落",false);
            _ghostType = 2;
         }
      }
      
      private function startGhostStep() : Boolean
      {
         if(_fighter.qi < 60)
         {
            return false;
         }
         if(!_fighter.hasEnergy(80,true))
         {
            return false;
         }
         _fighter.useQi(60);
         _fighter.useEnergy(80);
         _fighter.getCtrler().setDirectToTarget();
         _ghostStepIng = true;
         _ghostStepFrame = 12;
         _fighter.isAllowBeHit = false;
         _fighter.isCross = true;
         effectCtrler.ghostStep();
         _isContinuousGhostStep = false;
         return true;
      }
      
      private function renderGhostStep() : void
      {
         var _loc1_:Number = NaN;
         if(_ghostStepFrame-- <= 0)
         {
            if(_ghostType == 1)
            {
               _loc1_ = _fighter.getVecY();
               _action.isJumping = false;
               endGhostStep();
               _fighter.setVelocity(0,_loc1_);
               _fighter.setDamping(0,-_loc1_ / 10);
               setAirMove(true);
               return;
            }
            endGhostStep();
         }
         if(_ghostType == 2)
         {
            if(GameLogic.isTouchBottomFloor(_fighter))
            {
               endGhostStep();
            }
         }
      }
      
      private function endGhostStep() : void
      {
         if(_ghostStepFrame > 0)
         {
            _fighter.delayCall(recoverIsContinuousGhostStep,_ghostStepFrame);
         }
         else
         {
            recoverIsContinuousGhostStep();
         }
         _ghostStepIng = false;
         effectCtrler.endGhostStep();
         _fighter.getCtrler().setDirectToTarget();
         idle();
      }
      
      private function recoverIsContinuousGhostStep() : void
      {
         _isContinuousGhostStep = true;
      }
   }
}

