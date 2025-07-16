import 'dart:developer';

import 'package:bloc/bloc.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:meta/meta.dart';

part 'auth_state.dart';

class AuthCubit extends Cubit<AuthState> {
  AuthCubit() : super(AuthInitial());

  final FirebaseAuth _auth = FirebaseAuth.instance;

  Future<void> loginUser({required String email, required String password}) async {
    emit(LoginLoading());
    try {
      await _auth.signInWithEmailAndPassword(email: email, password: password);
      emit(LoginSuccess());
    } on FirebaseAuthException catch (e) {
      emit(LoginFailure(errorMessage: _getFirebaseErrorMessage(e.code)));
    } catch (e) {
      log('Unexpected error during login: $e');
      emit(LoginFailure(errorMessage: 'An unexpected error occurred.'));
    }
  }

  Future<void> registerUser({required String email, required String password}) async {
    emit(RegisterLoading());
    try {
      await _auth.createUserWithEmailAndPassword(email: email, password: password);
      emit(RegisterSuccess());
    } on FirebaseAuthException catch (e) {
      emit(RegisterFailure(errorMessage: _getFirebaseErrorMessage(e.code)));
    } catch (e) {
      log('Unexpected error during registration: $e');
      emit(RegisterFailure(errorMessage: 'An unexpected error occurred.'));
    }
  }

  String _getFirebaseErrorMessage(String errorCode) {
    switch (errorCode) {
      case 'invalid-email':
        return 'Invalid email format.';
      case 'user-not-found':
        return 'No user found for this email.';
      case 'wrong-password':
        return 'Incorrect password.';
      case 'email-already-in-use':
        return 'Email is already in use.';
      case 'weak-password':
        return 'Password is too weak.';
      default:
        return 'Something went wrong. Please try again.';
    }
  }
}
